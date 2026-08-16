from __future__ import annotations

import argparse, json, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baselines import BASELINES, evaluate_baseline
from mac_env import SCENARIOS, SimConfig
from pdppo import PPOConfig, TRAINING_SCENARIOS, evaluate_agent, save_agent, train_agent

RL_MODES=["unconstrained","fixed_penalty","single_constraint","multi_constraint"]
METHOD_LABEL={
 "unconstrained":"ReferenceResidual-Unconstrained-PPO",
 "fixed_penalty":"ReferenceResidual-FixedPenalty-PPO",
 "single_constraint":"ReferenceResidual-SingleConstraint-PD-PPO",
 "multi_constraint":"ReferenceResidual-MultiConstraint-PD-PPO",
}


def parse_args():
 p=argparse.ArgumentParser(description="MAC Scheduler V2.6 reference-anchored residual hierarchical experiments")
 p.add_argument("--profile",choices=["smoke","pilot","paper"],default="pilot")
 p.add_argument("--steps",type=int,default=None); p.add_argument("--eval-steps",type=int,default=None)
 p.add_argument("--seeds",type=int,default=None); p.add_argument("--device",default="auto"); p.add_argument("--out",default="results_v2_6")
 p.add_argument("--methods",nargs="*",default=None,help="PF MT EDF DeadlineAwarePF SLAWeightedPF ReferenceClassPF StaticClassPF DemandAwareClassPF unconstrained fixed_penalty single_constraint multi_constraint")
 p.add_argument("--scenarios",nargs="*",default=None)
 return p.parse_args()


def profile_values(p):
 return (4096,1500,1) if p=="smoke" else ((100_000,6000,3) if p=="pilot" else (800_000,10_000,5))


def summarize(df):
 metrics=["throughput_mbps","embb_throughput_mbps","urllc_throughput_mbps","center_embb_throughput_mbps","edge_embb_throughput_mbps",
 "min_embb_throughput_mbps","min_embb_satisfaction_ratio","mean_embb_satisfaction_ratio","latency_violation","rate_deficit_cost","rate_violation_fraction","drop_ratio",
 "packet_deadline_miss_ratio","worst_urllc_deadline_miss_ratio","urllc_delay_p95_ms","urllc_delay_p99_ms","jain_fairness","avg_prb_embb","avg_prb_urllc",
 "avg_safety_reserved_prbs","safety_activation_fraction","safety_cap_hit_fraction","avg_urllc_class_share","std_urllc_class_share","avg_embb_sla_weight",
 "avg_reference_urllc_share","avg_reference_embb_weight","avg_delta_rho","std_delta_rho","avg_delta_beta","std_delta_beta","inference_us"]
 rows=[]
 for (m,s),g in df.groupby(["method","scenario"],sort=False):
  r={"method":m,"scenario":s,"n":len(g)}
  for x in metrics:
   if x in g.columns:
    r[x+"_mean"]=g[x].mean(); r[x+"_std"]=g[x].std(ddof=1) if len(g)>1 else 0.0
  rows.append(r)
 return pd.DataFrame(rows)


def plots(summary,out):
 d=out/"figures"; d.mkdir(parents=True,exist_ok=True)
 order=["nominal_hetero","load_surge","edge_fade","bursty_urllc","mixed_dynamic","hotspot_unseen"]
 dyn=summary[summary.scenario.isin(order)]
 if dyn.empty:return
 x=np.arange(len(order))
 for metric,ylabel,fname in [("throughput_mbps_mean","Throughput (Mbps)","dynamic_throughput.png"),("packet_deadline_miss_ratio_mean","Packet deadline-miss ratio","dynamic_packet_miss.png"),("min_embb_satisfaction_ratio_mean","Worst eMBB throughput / requirement","dynamic_embb_satisfaction.png")]:
  fig,ax=plt.subplots(figsize=(9.5,5.0))
  for m in dyn.method.unique():
   g=dyn[dyn.method==m].set_index("scenario"); y=[g.loc[s,metric] if s in g.index else np.nan for s in order]
   ax.plot(x,y,marker="o",label=m)
  ax.set_xticks(x,[s.replace("_"," ") for s in order],rotation=20); ax.set_ylabel(ylabel); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(d/fname,dpi=220); plt.close(fig)
 # Learned/reference class share shows the reviewer-requested cross-service tradeoff directly.
 if "avg_urllc_class_share_mean" in dyn.columns:
  fig,ax=plt.subplots(figsize=(9.5,5.0))
  for m in dyn.method.unique():
   g=dyn[dyn.method==m].set_index("scenario"); y=[g.loc[s,"avg_urllc_class_share_mean"] if s in g.index else np.nan for s in order]
   if np.all(np.isnan(y)): continue
   ax.plot(x,y,marker="o",label=m)
  ax.set_xticks(x,[s.replace("_"," ") for s in order],rotation=20); ax.set_ylabel("Average residual URLLC class share"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(d/"learned_class_share.png",dpi=220); plt.close(fig)


 if "avg_delta_rho_mean" in dyn.columns:
   fig,ax=plt.subplots(figsize=(9.5,5.0))
   for m in dyn.method.unique():
    g=dyn[dyn.method==m].set_index("scenario"); y=[g.loc[s,"avg_delta_rho_mean"] if s in g.index else np.nan for s in order]
    if np.all(np.isnan(y)): continue
    ax.plot(x,y,marker="o",label=m)
   ax.axhline(0.0,linewidth=1)
   ax.set_xticks(x,[s.replace("_"," ") for s in order],rotation=20); ax.set_ylabel(r"Learned residual class-share correction $\Delta\rho$"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(d/"learned_delta_rho.png",dpi=220); plt.close(fig)


def main():
 a=parse_args(); ds,de,dn=profile_values(a.profile); steps=a.steps or ds; eval_steps=a.eval_steps or de; nseeds=a.seeds or dn
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"models").mkdir(exist_ok=True)
 sim=SimConfig(episode_steps=eval_steps); ppo=PPOConfig(total_steps=steps)
 if a.profile=="smoke": ppo.rollout_steps=1024; ppo.update_epochs=2; ppo.minibatch_size=128; ppo.hidden_size=128
 elif a.profile=="pilot": ppo.rollout_steps=2048; ppo.update_epochs=5
 allm=list(BASELINES)+RL_MODES; methods=a.methods or allm; scenarios=a.scenarios or list(SCENARIOS)
 badm=sorted(set(methods)-set(allm)); bads=sorted(set(scenarios)-set(SCENARIOS))
 if badm or bads: raise SystemExit(f"Unknown methods={badm} scenarios={bads}")
 with open(out/"config.json","w") as f: json.dump({"version":"V2.6-reference-anchored-residual","policy_structure":"2-D bounded residual PPO action: delta-rho + delta-beta around a state-aware deadline reference; explicit global constraint-pressure features; deterministic intra-class scheduling + minimal emergency shield","sim":asdict(sim),"ppo":asdict(ppo),"training_scenarios":[asdict(x) for x in TRAINING_SCENARIOS],"methods":methods,"scenarios":scenarios},f,indent=2)
 rows=[]; hist=[]; t0=time.time()
 for m in methods:
  if m not in BASELINES: continue
  for seed in range(nseeds):
   base=10000+seed*100
   for j,s in enumerate(scenarios):
    r=evaluate_baseline(sim,m,base+j,s); rows.append({"method":m,"scenario":s,"seed":seed,**r})
    print(f"{m:20s} seed={seed} scenario={s:17s} thr={r['throughput_mbps']:.2f} hol={r['latency_violation']:.4f} rdef={r['rate_deficit_cost']:.4f} miss={r['packet_deadline_miss_ratio']:.4f} rho={r.get('avg_urllc_class_share',np.nan):.3f} ref={r.get('avg_reference_urllc_share',np.nan):.3f} drho={r.get('avg_delta_rho',np.nan):+.3f}")
 for mode in RL_MODES:
  if mode not in methods: continue
  label=METHOD_LABEL[mode]
  for seed in range(nseeds):
   tsim=replace(sim,episode_steps=max(3000,min(eval_steps,5000))); agent,h=train_agent(tsim,ppo,seed,mode,a.device,True); save_agent(agent,out/"models"/f"{mode}_seed{seed}.pt",tsim,ppo)
   hist += [{"method":label,"seed":seed,**x} for x in h]; base=10000+seed*100
   for j,s in enumerate(scenarios):
    r=evaluate_agent(agent,sim,base+j,s); rows.append({"method":label,"scenario":s,"seed":seed,**r})
    print(f"{label:38s} seed={seed} scenario={s:17s} thr={r['throughput_mbps']:.2f} hol={r['latency_violation']:.4f} rdef={r['rate_deficit_cost']:.4f} miss={r['packet_deadline_miss_ratio']:.4f} rho={r['avg_urllc_class_share']:.3f} ref={r['avg_reference_urllc_share']:.3f} drho={r['avg_delta_rho']:+.3f} beta={r['avg_embb_sla_weight']:.2f}")
 raw=pd.DataFrame(rows); raw.to_csv(out/"raw_results.csv",index=False); summ=summarize(raw); summ.to_csv(out/"summary.csv",index=False)
 if hist: pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False)
 plots(summ,out); print(f"\nFinished in {(time.time()-t0)/60:.1f} min\nResults: {out}")

if __name__=="__main__": main()
