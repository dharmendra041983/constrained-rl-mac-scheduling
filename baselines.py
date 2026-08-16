from __future__ import annotations

import time
from typing import Dict, Type
import numpy as np

from mac_env import CQI_EFF, MacSchedulingEnv, SimConfig
from hierarchy import allocate_hierarchical, allocate_reference_residual, demand_aware_controls, largest_remainder


def _to_logits(prbs: np.ndarray, n_prb: int) -> np.ndarray:
    frac = (np.asarray(prbs, dtype=np.float64) + 1e-9) / (max(1, n_prb) + 1e-9)
    return np.log(frac + 1e-12)


class SchedulerBase:
    hierarchical = False
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.avg_rate = np.full(cfg.n_ue, 1e-3, dtype=np.float64)
    def update(self, sent_bytes: np.ndarray):
        a = 0.02
        self.avg_rate = (1 - a) * self.avg_rate + a * np.asarray(sent_bytes, dtype=np.float64)
    def allocate(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class ProportionalFair(SchedulerBase):
    def allocate(self, state):
        nonempty = state["q_bytes"] > 0
        eff = CQI_EFF[state["cqi"].astype(int)]
        w = np.where(nonempty, eff / (self.avg_rate + 1e-6), 0.0)
        return largest_remainder(w, self.cfg.n_prb)


class MaxThroughput(SchedulerBase):
    def allocate(self, state):
        nonempty = state["q_bytes"] > 0
        score = np.where(nonempty, CQI_EFF[state["cqi"].astype(int)], -1.0)
        out = np.zeros(self.cfg.n_ue, dtype=np.int64)
        if np.any(nonempty): out[int(np.argmax(score))] = self.cfg.n_prb
        return out


class EarliestDeadlineFirst(SchedulerBase):
    def allocate(self, state):
        out = np.zeros(self.cfg.n_ue, dtype=np.int64)
        nonempty = state["q_bytes"] > 0
        u = np.asarray(self.cfg.urllc_ues, dtype=int)
        if u.size and np.any(nonempty[u]):
            urgency = state["hol_ms"][u] / np.maximum(state["deadlines_ms"][u], 1e-6)
            out[int(u[np.argmax(np.where(nonempty[u], urgency, -1.0))])] = self.cfg.n_prb
        elif np.any(nonempty):
            eff = np.where(nonempty, CQI_EFF[state["cqi"].astype(int)], -1.0)
            out[int(np.argmax(eff))] = self.cfg.n_prb
        return out


class DeadlineAwarePF(SchedulerBase):
    def allocate(self, state):
        nonempty = state["q_bytes"] > 0
        out = np.zeros(self.cfg.n_ue, dtype=np.int64)
        deadlines = state["deadlines_ms"]
        urllc = list(self.cfg.urllc_ues)
        urllc.sort(key=lambda u: float(state["hol_ms"][u] / max(deadlines[u], 1e-6)), reverse=True)
        for ue in urllc:
            if not nonempty[ue]: continue
            eff = float(CQI_EFF[int(state["cqi"][ue])])
            bpp = max(1.0, eff * self.cfg.re_per_prb * self.cfg.phy_overhead_factor / 8.0)
            need = int(np.ceil(float(state["q_bytes"][ue]) / bpp))
            give = min(need, self.cfg.n_prb - int(out.sum()))
            out[ue] += max(0, give)
            if out.sum() >= self.cfg.n_prb: return out
        rem = self.cfg.n_prb - int(out.sum())
        if rem > 0:
            eff = CQI_EFF[state["cqi"].astype(int)]
            pf = np.where(nonempty, eff / (self.avg_rate + 1e-6), 0.0)
            pf[self.cfg.urllc_ues] = 0.0
            if pf.sum() > 0: out += largest_remainder(pf, rem)
            elif np.any(nonempty): out[int(np.argmax(np.where(nonempty, eff, -1.0)))] += rem
        return out


class SLAWeightedPF(SchedulerBase):
    def allocate(self, state):
        nonempty = state["q_bytes"] > 0
        out = np.zeros(self.cfg.n_ue, dtype=np.int64)
        deadlines = state["deadlines_ms"]
        urllc = list(self.cfg.urllc_ues)
        urllc.sort(key=lambda u: float(state["hol_ms"][u] / max(deadlines[u], 1e-6)), reverse=True)
        for ue in urllc:
            if not nonempty[ue]: continue
            urgency = float(state["hol_ms"][ue] / max(deadlines[ue], 1e-6))
            eff = float(CQI_EFF[int(state["cqi"][ue])])
            bpp = max(1.0, eff * self.cfg.re_per_prb * self.cfg.phy_overhead_factor / 8.0)
            target = float(state["q_bytes"][ue]) * min(1.0, 0.50 + 0.75 * urgency)
            need = max(1, int(np.ceil(target / bpp)))
            give = min(need, self.cfg.n_prb - int(out.sum()))
            out[ue] += give
            if out.sum() >= self.cfg.n_prb: return out
        rem = self.cfg.n_prb - int(out.sum())
        if rem > 0:
            eff = CQI_EFF[state["cqi"].astype(int)]
            w = np.zeros(self.cfg.n_ue, dtype=np.float64)
            for ue in self.cfg.embb_ues:
                if not nonempty[ue]: continue
                target = max(1e-6, float(state["min_rates_mbps"][ue]))
                deficit = max(0.0, target - float(state["avg_rate_mbps"][ue])) / target
                w[ue] = eff[ue] / (self.avg_rate[ue] + 1e-6) * (1 + 3.0 * deficit) * (1.15 if state["edge_mask"][ue] else 1.0)
            if w.sum() > 0: out += largest_remainder(w, rem)
        return out


class ReferenceClassPF(SchedulerBase):
    hierarchical = True
    def allocate_components(self, state):
        # Zero residual action: execute the exact V2.6 state-aware reference.
        return allocate_reference_residual(self.cfg, state, np.zeros(2, dtype=np.float32), use_shield=True)


class StaticClassPF(SchedulerBase):
    hierarchical = True
    def controls(self, state):
        return float(self.cfg.static_rho), float(self.cfg.static_beta)
    def allocate_components(self, state):
        rho, beta = self.controls(state)
        return allocate_hierarchical(self.cfg, state, rho, beta, use_shield=True)


class DemandAwareClassPF(StaticClassPF):
    def controls(self, state):
        return demand_aware_controls(self.cfg, state)


BASELINES: Dict[str, Type[SchedulerBase]] = {
    "PF": ProportionalFair,
    "MT": MaxThroughput,
    "EDF": EarliestDeadlineFirst,
    "DeadlineAwarePF": DeadlineAwarePF,
    "SLAWeightedPF": SLAWeightedPF,
    "ReferenceClassPF": ReferenceClassPF,
    "StaticClassPF": StaticClassPF,
    "DemandAwareClassPF": DemandAwareClassPF,
}


def evaluate_baseline(cfg: SimConfig, scheduler_name: str, seed: int, scenario: str) -> Dict[str, float]:
    env = MacSchedulingEnv(cfg, seed=seed, scenario=scenario)
    sched = BASELINES[scheduler_name](cfg)
    env.reset(seed)
    latencies, rhos, betas, rho_refs, beta_refs, delta_rhos, delta_betas = [], [], [], [], [], [], []
    for _ in range(cfg.episode_steps):
        state = env.raw_state()
        t0 = time.perf_counter_ns()
        if getattr(sched, "hierarchical", False):
            reserved, residual, meta = sched.allocate_components(state)
            logits = _to_logits(residual, max(1, cfg.n_prb - int(reserved.sum())))
            latencies.append((time.perf_counter_ns() - t0) / 1000.0)
            _, _, _, done, info = env.step(logits, reserved_prbs=reserved, safety_info=meta)
            rhos.append(float(meta["rho"])); betas.append(float(meta["beta"]))
            rho_refs.append(float(meta.get("rho_ref", np.nan))); beta_refs.append(float(meta.get("beta_ref", np.nan)))
            delta_rhos.append(float(meta.get("delta_rho", np.nan))); delta_betas.append(float(meta.get("delta_beta", np.nan)))
        else:
            prbs = sched.allocate(state)
            latencies.append((time.perf_counter_ns() - t0) / 1000.0)
            _, _, _, done, info = env.step(_to_logits(prbs, cfg.n_prb))
        sched.update(info["sent_bytes"])
        if done: break
    out = env.metrics()
    out["inference_us"] = float(np.median(latencies)) if latencies else 0.0
    out["avg_urllc_class_share"] = float(np.mean(rhos)) if rhos else np.nan
    out["avg_embb_sla_weight"] = float(np.mean(betas)) if betas else np.nan
    out["avg_reference_urllc_share"] = float(np.nanmean(rho_refs)) if rho_refs and not np.all(np.isnan(rho_refs)) else np.nan
    out["avg_reference_embb_weight"] = float(np.nanmean(beta_refs)) if beta_refs and not np.all(np.isnan(beta_refs)) else np.nan
    out["avg_delta_rho"] = float(np.nanmean(delta_rhos)) if delta_rhos and not np.all(np.isnan(delta_rhos)) else np.nan
    out["avg_delta_beta"] = float(np.nanmean(delta_betas)) if delta_betas and not np.all(np.isnan(delta_betas)) else np.nan
    return out
