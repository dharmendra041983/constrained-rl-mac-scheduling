from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from hierarchy import allocate_reference_residual
from mac_env import MacSchedulingEnv, SimConfig, Scenario


@dataclass
class PPOConfig:
    total_steps: int = 800_000
    rollout_steps: int = 4096
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    entropy_coef: float = 0.0015
    max_grad_norm: float = 0.5
    hidden_size: int = 192

    dual_lr_lat: float = 0.80
    dual_lr_rate: float = 0.55
    dual_lr_drop: float = 0.55
    lambda_max: float = 50.0
    initial_log_std: float = -1.15
    actor_mean_l2: float = 6e-4
    fixed_penalties: Tuple[float, float, float] = (5.0, 4.0, 3.0)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01 if out_dim > 1 else 1.0)

    def forward(self, x): return self.net(x)


class ActorMultiCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden: int = 192, n_costs: int = 3, initial_log_std: float = -1.15):
        super().__init__()
        self.actor = MLP(obs_dim, action_dim, hidden)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        self.reward_critic = MLP(obs_dim, 1, hidden)
        self.cost_critics = nn.ModuleList([MLP(obs_dim, 1, hidden) for _ in range(n_costs)])

    def dist(self, obs):
        return Normal(self.actor(obs), torch.exp(torch.clamp(self.log_std, -5.0, 1.0)))

    def values(self, obs):
        rv = self.reward_critic(obs).squeeze(-1)
        cv = torch.stack([c(obs).squeeze(-1) for c in self.cost_critics], dim=-1)
        return rv, cv

    def act(self, obs, deterministic=False):
        d = self.dist(obs); a = d.mean if deterministic else d.sample()
        rv, cv = self.values(obs)
        return a, d.log_prob(a).sum(-1), rv, cv


@dataclass
class TrainedAgent:
    model: ActorMultiCritic
    lambdas: np.ndarray
    device: torch.device
    mode: str

    def raw_action(self, obs: np.ndarray, deterministic=True) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a, _, _, _ = self.model.act(x, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()


def _to_logits(prbs: np.ndarray, budget: int) -> np.ndarray:
    x = np.asarray(prbs, dtype=np.float64) + 1e-9
    x = x / max(x.sum(), 1e-12)
    return np.log(np.clip(x, 1e-12, 1.0)).astype(np.float32)


def _gae(rewards, values, dones, last_value, gamma, lam):
    n = len(rewards); adv = np.zeros(n, dtype=np.float32); last_gae = 0.0; nv = float(last_value)
    for t in reversed(range(n)):
        mask = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * nv * mask - float(values[t])
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae; nv = float(values[t])
    return adv, adv + np.asarray(values, dtype=np.float32)


def _mode_setup(mode: str, ppo: PPOConfig):
    if mode == "unconstrained": return np.zeros(3, np.float32), np.zeros(3, bool), False
    if mode == "fixed_penalty": return np.asarray(ppo.fixed_penalties, np.float32), np.ones(3, bool), False
    if mode == "single_constraint": return np.array([1.,0.,0.],np.float32), np.array([1,0,0],bool), True
    if mode == "multi_constraint": return np.array([1.,0.5,0.5],np.float32), np.ones(3,bool), True
    raise ValueError(mode)


# Balanced curriculum; hotspot_unseen and tight_sla stay evaluation-only.
TRAINING_SCENARIOS = [
    Scenario("train_nominal"),
    Scenario("train_load", surge_scale=1.35, surge_start=0.28, surge_end=0.80),
    Scenario("train_edge", fade_ues=(4,5,8,9), fade_shift=-3, fade_start=0.35, fade_end=1.0),
    Scenario("train_burst", burst_scale=1.90, burst_start=0.28, burst_end=0.82),
    Scenario("train_joint", surge_scale=1.28, surge_start=0.25, surge_end=0.80,
             burst_scale=1.55, burst_start=0.42, burst_end=0.90,
             fade_ues=(4,5,8,9), fade_shift=-2, fade_start=0.40, fade_end=1.0),
]


def _randomize_training_scenario(template: Scenario, rng: np.random.Generator) -> Scenario:
    if template.name == "train_nominal":
        return replace(template, load_scale=float(rng.uniform(.96,1.06)), urllc_scale=float(rng.uniform(.95,1.08)),
                       global_cqi_shift=int(rng.choice([0,0,0,-1])))
    if template.name == "train_load": return replace(template, surge_scale=float(template.surge_scale*rng.uniform(.90,1.12)))
    if template.name == "train_edge": return replace(template, fade_shift=int(rng.choice([-2,-3,-3])))
    if template.name == "train_burst": return replace(template, burst_scale=float(template.burst_scale*rng.uniform(.88,1.12)))
    if template.name == "train_joint":
        return replace(template, surge_scale=float(template.surge_scale*rng.uniform(.92,1.10)),
                       burst_scale=float(template.burst_scale*rng.uniform(.90,1.12)), fade_shift=int(rng.choice([-2,-2,-3])))
    return template


def train_agent(sim: SimConfig, ppo: PPOConfig, seed: int, mode="multi_constraint", device="auto", verbose=True):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if device == "auto": dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: dev = torch.device(device)

    srng = np.random.default_rng(seed + 9137); order=[]; cursor=0
    def next_scenario():
        nonlocal order, cursor
        if cursor >= len(order): order=srng.permutation(len(TRAINING_SCENARIOS)).tolist(); cursor=0
        x = _randomize_training_scenario(TRAINING_SCENARIOS[int(order[cursor])], srng); cursor += 1; return x

    env = MacSchedulingEnv(sim, seed=seed, scenario=next_scenario()); obs = env.reset(seed)
    model = ActorMultiCritic(env.obs_dim, 2, ppo.hidden_size, initial_log_std=ppo.initial_log_std).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)
    lambdas, active, adaptive = _mode_setup(mode, ppo)
    dual_lrs = np.array([ppo.dual_lr_lat, ppo.dual_lr_rate, ppo.dual_lr_drop], np.float32)
    limits = sim.cost_limits
    history: List[Dict[str,float]]=[]; global_step=0; rollout_index=0

    while global_step < ppo.total_steps:
        n = min(ppo.rollout_steps, ppo.total_steps-global_step)
        ob=np.zeros((n,env.obs_dim),np.float32); ac=np.zeros((n,2),np.float32); lp=np.zeros(n,np.float32)
        rw=np.zeros(n,np.float32); co=np.zeros((n,3),np.float32); dn=np.zeros(n,np.float32)
        rvb=np.zeros(n,np.float32); cvb=np.zeros((n,3),np.float32); rhos=[]; betas=[]
        rho_refs=[]; beta_refs=[]; delta_rhos=[]; delta_betas=[]
        scount={x.name:0 for x in TRAINING_SCENARIOS}
        for t in range(n):
            x=torch.as_tensor(obs,dtype=torch.float32,device=dev).unsqueeze(0)
            with torch.no_grad(): a, logp, rv, cv = model.act(x, deterministic=False)
            raw=a.squeeze(0).cpu().numpy()
            reserved,residual,meta=allocate_reference_residual(sim,env.raw_state(),raw,use_shield=True)
            rho=float(meta["rho"]); beta=float(meta["beta"])
            next_obs,reward,costs,done,info=env.step(_to_logits(residual,max(1,sim.n_prb-int(reserved.sum()))),
                                                     reserved_prbs=reserved,safety_info=meta)
            ob[t]=obs; ac[t]=raw; lp[t]=float(logp.item()); rw[t]=reward; co[t]=costs; dn[t]=float(done)
            rvb[t]=float(rv.item()); cvb[t]=cv.squeeze(0).cpu().numpy(); rhos.append(rho); betas.append(beta)
            rho_refs.append(float(meta["rho_ref"])); beta_refs.append(float(meta["beta_ref"]))
            delta_rhos.append(float(meta["delta_rho"])); delta_betas.append(float(meta["delta_beta"]))
            scount[env.scenario.name]=scount.get(env.scenario.name,0)+1
            obs=next_obs; global_step += 1
            if done:
                env=MacSchedulingEnv(sim,seed=seed+global_step+17,scenario=next_scenario()); obs=env.reset(seed+global_step+17)

        with torch.no_grad():
            x=torch.as_tensor(obs,dtype=torch.float32,device=dev).unsqueeze(0); last_rv,last_cv=model.values(x)
            last_rv=float(last_rv.item()); last_cv=last_cv.squeeze(0).cpu().numpy()
        ua,uret=_gae(rw,rvb,dn,last_rv,ppo.gamma,ppo.gae_lambda)
        ca=np.zeros_like(co); cret=np.zeros_like(co)
        for k in range(3): ca[:,k],cret[:,k]=_gae(co[:,k],cvb[:,k],dn,last_cv[k],ppo.gamma,ppo.gae_lambda)
        ua=(ua-ua.mean())/(ua.std()+1e-8)
        for k in range(3): ca[:,k]=(ca[:,k]-ca[:,k].mean())/(ca[:,k].std()+1e-8)
        el=lambdas*active.astype(np.float32); adv=ua-np.sum(ca*el[None,:],axis=1); adv/=1+float(el.sum())
        adv=(adv-adv.mean())/(adv.std()+1e-8)

        obt=torch.as_tensor(ob,dtype=torch.float32,device=dev); act=torch.as_tensor(ac,dtype=torch.float32,device=dev)
        oldlp=torch.as_tensor(lp,dtype=torch.float32,device=dev); adt=torch.as_tensor(adv,dtype=torch.float32,device=dev)
        urt=torch.as_tensor(uret,dtype=torch.float32,device=dev); crt=torch.as_tensor(cret,dtype=torch.float32,device=dev)
        idx=np.arange(n); losses=[]
        for _ in range(ppo.update_epochs):
            np.random.shuffle(idx)
            for st in range(0,n,ppo.minibatch_size):
                mb=idx[st:st+ppo.minibatch_size]; mt=torch.as_tensor(mb,dtype=torch.long,device=dev)
                d=model.dist(obt[mt]); nlp=d.log_prob(act[mt]).sum(-1); ent=d.entropy().sum(-1).mean()
                ratio=torch.exp(nlp-oldlp[mt]); aa=adt[mt]
                actor=-torch.min(ratio*aa,torch.clamp(ratio,1-ppo.clip_ratio,1+ppo.clip_ratio)*aa).mean()
                vr,vc=model.values(obt[mt]); vl=((vr-urt[mt])**2).mean(); cl=((vc-crt[mt])**2).mean()
                mean_l2=torch.mean(model.actor(obt[mt])**2)
                loss=actor+ppo.value_coef*vl+ppo.cost_value_coef*cl+ppo.actor_mean_l2*mean_l2-ppo.entropy_coef*ent
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),ppo.max_grad_norm); opt.step()
                losses.append(float(loss.item()))
        mc=co.mean(axis=0)
        if adaptive:
            for k in range(3):
                lambdas[k]=np.clip(lambdas[k]+dual_lrs[k]*(mc[k]-limits[k]),0,ppo.lambda_max) if active[k] else 0.0
        history.append({"rollout":rollout_index,"step":global_step,"reward_mean":float(rw.mean()),
                        "lat_cost":float(mc[0]),"rate_cost":float(mc[1]),"drop_cost":float(mc[2]),
                        "lambda_lat":float(lambdas[0]),"lambda_rate":float(lambdas[1]),"lambda_drop":float(lambdas[2]),
                        "rho_mean":float(np.mean(rhos)),"rho_std":float(np.std(rhos)),"beta_mean":float(np.mean(betas)),
                        "rho_ref_mean":float(np.mean(rho_refs)),"beta_ref_mean":float(np.mean(beta_refs)),
                        "delta_rho_mean":float(np.mean(delta_rhos)),"delta_rho_std":float(np.std(delta_rhos)),
                        "delta_beta_mean":float(np.mean(delta_betas)),"delta_beta_std":float(np.std(delta_betas)),
                        "policy_log_std_mean":float(model.log_std.detach().mean().cpu()),"loss":float(np.mean(losses)),
                        **{f"steps_{k}":int(v) for k,v in scount.items()}})
        rollout_index+=1
        if verbose and (rollout_index==1 or rollout_index%10==0 or global_step>=ppo.total_steps):
            h=history[-1]; print(f"[{mode}] step={global_step} reward={h['reward_mean']:.3f} costs=({h['lat_cost']:.3f},{h['rate_cost']:.3f},{h['drop_cost']:.3f}) lambda=({h['lambda_lat']:.2f},{h['lambda_rate']:.2f},{h['lambda_drop']:.2f}) rho={h['rho_mean']:.3f} ref={h['rho_ref_mean']:.3f} drho={h['delta_rho_mean']:+.3f} beta={h['beta_mean']:.2f}")

    return TrainedAgent(model,lambdas.copy(),dev,mode), history


def evaluate_agent(agent: TrainedAgent, sim: SimConfig, seed: int, scenario: str) -> Dict[str,float]:
    env=MacSchedulingEnv(sim,seed=seed,scenario=scenario); obs=env.reset(seed)
    lat=[]; rhos=[]; betas=[]; rho_refs=[]; beta_refs=[]; delta_rhos=[]; delta_betas=[]
    for _ in range(sim.episode_steps):
        if agent.device.type=="cuda": torch.cuda.synchronize()
        t0=time.perf_counter_ns(); raw=agent.raw_action(obs,True)
        reserved,residual,meta=allocate_reference_residual(sim,env.raw_state(),raw,use_shield=True)
        if agent.device.type=="cuda": torch.cuda.synchronize()
        lat.append((time.perf_counter_ns()-t0)/1000.0)
        rhos.append(float(meta["rho"])); betas.append(float(meta["beta"]))
        rho_refs.append(float(meta["rho_ref"])); beta_refs.append(float(meta["beta_ref"]))
        delta_rhos.append(float(meta["delta_rho"])); delta_betas.append(float(meta["delta_beta"]))
        obs,_,_,done,_=env.step(_to_logits(residual,max(1,sim.n_prb-int(reserved.sum()))),reserved_prbs=reserved,safety_info=meta)
        if done: break
    out=env.metrics(); out["inference_us"]=float(np.median(lat)); out["avg_urllc_class_share"]=float(np.mean(rhos)); out["std_urllc_class_share"]=float(np.std(rhos)); out["avg_embb_sla_weight"]=float(np.mean(betas))
    out["avg_reference_urllc_share"]=float(np.mean(rho_refs)); out["avg_reference_embb_weight"]=float(np.mean(beta_refs))
    out["avg_delta_rho"]=float(np.mean(delta_rhos)); out["std_delta_rho"]=float(np.std(delta_rhos)); out["avg_delta_beta"]=float(np.mean(delta_betas)); out["std_delta_beta"]=float(np.std(delta_betas))
    out["lambda_lat"]=float(agent.lambdas[0]); out["lambda_rate"]=float(agent.lambdas[1]); out["lambda_drop"]=float(agent.lambdas[2])
    return out


def save_agent(agent: TrainedAgent, path: str|Path, sim: SimConfig, ppo: PPOConfig):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":agent.model.state_dict(),"lambdas":agent.lambdas,"mode":agent.mode,
                "obs_dim":agent.model.actor.net[0].in_features,"action_dim":2,"hidden_size":ppo.hidden_size,
                "policy_structure":"V2.6 reference-anchored residual hierarchical PPO: deadline-aware reference class share + bounded learned delta-rho/delta-beta, deterministic intra-class schedulers, minimal emergency shield",
                "sim_config":sim.__dict__,"ppo_config":ppo.__dict__},path)
