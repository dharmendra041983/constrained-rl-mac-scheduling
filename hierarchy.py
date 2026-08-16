from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from mac_env import CQI_EFF, SimConfig
from shield import emergency_shield


def largest_remainder(weights: np.ndarray, budget: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros_like(w, dtype=np.int64)
    budget = int(max(0, budget))
    if budget <= 0 or np.sum(w) <= 0:
        return out
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-12)
    raw = w * budget
    out = np.floor(raw).astype(np.int64)
    rem = budget - int(out.sum())
    if rem > 0:
        out[np.argsort(-(raw - out))[:rem]] += 1
    return out


def capped_weighted_allocation(weights: np.ndarray, caps: np.ndarray, budget: int) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    caps = np.maximum(np.asarray(caps, dtype=np.int64), 0)
    out = np.zeros_like(caps, dtype=np.int64)
    remaining = int(max(0, budget))
    active = (weights > 0) & (caps > 0)
    while remaining > 0 and np.any(active):
        idx = np.where(active)[0]
        share = largest_remainder(weights[idx], remaining)
        if share.sum() <= 0:
            share[np.argmax(weights[idx])] = remaining
        room = caps[idx] - out[idx]
        give = np.minimum(share, room)
        if give.sum() <= 0:
            break
        out[idx] += give
        remaining -= int(give.sum())
        active = (weights > 0) & (out < caps)
    return out


def _bytes_per_prb(cfg: SimConfig, cqi: int) -> float:
    eff = float(CQI_EFF[int(cqi)])
    return max(1.0, eff * cfg.re_per_prb * cfg.phy_overhead_factor / 8.0)


def residual_from_raw(raw_action: np.ndarray, cfg: SimConfig) -> Tuple[float, float]:
    """Bound Gaussian policy outputs to residual corrections around a reference.

    Zero network output means zero correction, so the initial policy stays close
    to the strong state-aware reference instead of relearning class budgeting.
    """
    a = np.asarray(raw_action, dtype=np.float64).reshape(2)
    delta_rho = float(cfg.delta_rho_max * np.tanh(a[0]))
    delta_beta = float(cfg.delta_beta_max * np.tanh(a[1]))
    return delta_rho, delta_beta


def deadline_reference_controls(
    cfg: SimConfig,
    state: Dict[str, np.ndarray],
    reserved_prbs: np.ndarray | None = None,
) -> Tuple[float, float, Dict[str, float]]:
    """State-aware reference class controls derived from deadline-aware demand.

    The URLLC reference share equals the residual share a DeadlineAwarePF-style
    policy would require to clear current URLLC queues, after accounting for the
    emergency shield. The eMBB reference weight responds continuously to current
    minimum-rate deficits. This is a *reference*, not a learned action.
    """
    q = np.asarray(state["q_bytes"], dtype=np.float64)
    cqi = np.asarray(state["cqi"], dtype=np.int64)
    hol = np.asarray(state["hol_ms"], dtype=np.float64)
    deadlines = np.asarray(state["deadlines_ms"], dtype=np.float64)
    avg_rate = np.asarray(state["avg_rate_mbps"], dtype=np.float64)
    min_rates = np.asarray(state["min_rates_mbps"], dtype=np.float64)

    reserved = np.zeros(cfg.n_ue, dtype=np.int64) if reserved_prbs is None else np.asarray(reserved_prbs, dtype=np.int64)
    remaining = max(0, int(cfg.n_prb - reserved.sum()))
    u_nonempty = any(q[u] > 0 for u in cfg.urllc_ues)
    e_nonempty = any(q[u] > 0 for u in cfg.embb_ues)

    ref_extra = np.zeros(cfg.n_ue, dtype=np.int64)
    if remaining > 0 and u_nonempty:
        # Exact deadline-aware ordering: highest normalized HOL first, then
        # allocate enough PRBs for each current URLLC queue while budget remains.
        order = sorted(
            cfg.urllc_ues,
            key=lambda u: float(hol[u] / max(deadlines[u], 1e-6)),
            reverse=True,
        )
        left = remaining
        for ue in order:
            if q[ue] <= 0 or left <= 0:
                continue
            need_total = int(np.ceil(q[ue] / _bytes_per_prb(cfg, int(cqi[ue]))))
            need_extra = max(0, need_total - int(reserved[ue]))
            give = min(need_extra, left)
            ref_extra[ue] = give
            left -= give

    if not u_nonempty:
        rho_ref = 0.0
    elif not e_nonempty:
        rho_ref = 1.0
    elif remaining <= 0:
        rho_ref = 0.0
    else:
        rho_ref = float(ref_extra[cfg.urllc_ues].sum() / remaining)
        rho_ref = float(np.clip(rho_ref, cfg.rho_min, cfg.rho_max))

    deficits = []
    for ue in cfg.embb_ues:
        target = max(1e-6, float(min_rates[ue]))
        deficits.append(max(0.0, target - float(avg_rate[ue])) / target)
    mean_def = float(np.mean(deficits)) if deficits else 0.0
    worst_def = float(np.max(deficits)) if deficits else 0.0
    beta_ref = float(np.clip(cfg.reference_beta_base + 3.5 * mean_def + 1.5 * worst_def, 0.0, cfg.beta_max))

    urgency = [float(np.clip(hol[u] / max(deadlines[u], 1e-6), 0.0, 3.0)) for u in cfg.urllc_ues if q[u] > 0]
    return rho_ref, beta_ref, {
        "rho_ref": rho_ref,
        "beta_ref": beta_ref,
        "reference_urllc_prbs": float(ref_extra[cfg.urllc_ues].sum()),
        "reference_mean_urgency": float(np.mean(urgency)) if urgency else 0.0,
        "reference_max_urgency": float(np.max(urgency)) if urgency else 0.0,
        "reference_mean_rate_deficit": mean_def,
        "reference_worst_rate_deficit": worst_def,
    }


def allocate_hierarchical(
    cfg: SimConfig,
    state: Dict[str, np.ndarray],
    rho: float,
    beta: float,
    use_shield: bool = True,
    reserved_override: np.ndarray | None = None,
    safety_info_override: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Allocate residual PRBs by class, then deterministically within each class."""
    if reserved_override is not None:
        reserved = np.asarray(reserved_override, dtype=np.int64).copy()
        sinfo = dict(safety_info_override or {})
        sinfo.setdefault("safety_reserved_prbs", float(reserved.sum()))
        sinfo.setdefault("safety_active", float(reserved.sum() > 0))
        sinfo.setdefault("safety_cap_hit", 0.0)
        sinfo.setdefault("safety_reserved_fraction", float(reserved.sum() / max(1, cfg.n_prb)))
    elif use_shield:
        reserved, sinfo = emergency_shield(cfg, state)
    else:
        reserved = np.zeros(cfg.n_ue, dtype=np.int64)
        sinfo = {"safety_reserved_prbs": 0.0, "safety_active": 0.0, "safety_cap_hit": 0.0,
                 "safety_reserved_fraction": 0.0}

    remaining = int(cfg.n_prb - reserved.sum())
    residual = np.zeros(cfg.n_ue, dtype=np.int64)
    if remaining <= 0:
        sinfo.update({"rho": float(rho), "beta": float(beta), "urllc_budget": 0.0, "embb_budget": 0.0})
        return reserved, residual, sinfo

    q = np.asarray(state["q_bytes"], dtype=np.float64)
    cqi = np.asarray(state["cqi"], dtype=np.int64)
    hol = np.asarray(state["hol_ms"], dtype=np.float64)
    deadlines = np.asarray(state["deadlines_ms"], dtype=np.float64)
    avg_rate = np.asarray(state["avg_rate_mbps"], dtype=np.float64)
    min_rates = np.asarray(state["min_rates_mbps"], dtype=np.float64)
    burst = np.asarray(state.get("burst_on", np.zeros(cfg.n_ue, dtype=bool)), dtype=bool)
    edge = np.asarray(state.get("edge_mask", np.zeros(cfg.n_ue, dtype=bool)), dtype=bool)

    uidx = np.asarray(cfg.urllc_ues, dtype=int)
    eidx = np.asarray(cfg.embb_ues, dtype=int)
    u_nonempty = q[uidx] > 0
    e_nonempty = q[eidx] > 0

    if not np.any(u_nonempty):
        u_budget = 0
    elif not np.any(e_nonempty):
        u_budget = remaining
    else:
        u_budget = int(np.clip(np.rint(float(rho) * remaining), 0, remaining))
    e_budget = remaining - u_budget

    u_weights = np.zeros(cfg.n_ue, dtype=np.float64)
    u_caps = np.zeros(cfg.n_ue, dtype=np.int64)
    for ue in cfg.urllc_ues:
        if q[ue] <= 0:
            continue
        bpp = _bytes_per_prb(cfg, int(cqi[ue]))
        need = max(1, int(np.ceil(q[ue] / bpp)))
        urgency = float(np.clip(hol[ue] / max(deadlines[ue], 1e-6), 0.0, 2.5))
        burst_mult = 1.35 if burst[ue] else 1.0
        queue_pkts = q[ue] / max(1.0, float(cfg.urllc_pkt_bytes[ue - cfg.n_embb]))
        u_weights[ue] = need * (0.30 + urgency ** 2.0) * (1.0 + 0.10 * np.log1p(queue_pkts)) * burst_mult
        u_caps[ue] = need
    u_alloc = capped_weighted_allocation(u_weights, u_caps, u_budget)
    residual += u_alloc

    unused_u = u_budget - int(u_alloc.sum())
    e_budget += max(0, unused_u)

    e_weights = np.zeros(cfg.n_ue, dtype=np.float64)
    for ue in cfg.embb_ues:
        if q[ue] <= 0:
            continue
        eff = max(1e-6, float(CQI_EFF[int(cqi[ue])]))
        target = max(1e-6, float(min_rates[ue]))
        deficit = max(0.0, target - float(avg_rate[ue])) / target
        pf = eff / max(0.15, float(avg_rate[ue]))
        edge_bonus = 1.08 if edge[ue] else 1.0
        e_weights[ue] = pf * (1.0 + float(beta) * deficit) * edge_bonus
    e_alloc = largest_remainder(e_weights, e_budget)
    residual += e_alloc

    unused = remaining - int(residual.sum())
    if unused > 0 and np.any(u_nonempty):
        extra_caps = np.maximum(0, u_caps - residual)
        residual += capped_weighted_allocation(u_weights, extra_caps, unused)
        unused = remaining - int(residual.sum())
    if unused > 0:
        active = np.where(q > 0)[0]
        if active.size:
            best = int(active[np.argmax(CQI_EFF[cqi[active]])])
            residual[best] += unused

    sinfo.update({
        "rho": float(rho),
        "beta": float(beta),
        "urllc_budget": float(u_budget),
        "embb_budget": float(e_budget),
    })
    return reserved, residual, sinfo


def allocate_reference_residual(
    cfg: SimConfig,
    state: Dict[str, np.ndarray],
    raw_action: np.ndarray,
    use_shield: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Reference-anchored hierarchical action used by V2.6 PPO."""
    if use_shield:
        reserved, sinfo = emergency_shield(cfg, state)
    else:
        reserved = np.zeros(cfg.n_ue, dtype=np.int64)
        sinfo = {"safety_reserved_prbs": 0.0, "safety_active": 0.0, "safety_cap_hit": 0.0,
                 "safety_reserved_fraction": 0.0}
    rho_ref, beta_ref, refmeta = deadline_reference_controls(cfg, state, reserved)
    delta_rho_cmd, delta_beta_cmd = residual_from_raw(raw_action, cfg)
    q = np.asarray(state["q_bytes"], dtype=np.float64)
    u_nonempty = any(q[u] > 0 for u in cfg.urllc_ues)
    e_nonempty = any(q[u] > 0 for u in cfg.embb_ues)
    # When only one class is active, the class share is forced to 0 or 1 and
    # there is no meaningful residual correction to learn/log.
    if u_nonempty and e_nonempty:
        rho = float(np.clip(rho_ref + delta_rho_cmd, cfg.rho_min, cfg.rho_max))
    else:
        rho = float(rho_ref)
    beta = float(np.clip(beta_ref + delta_beta_cmd, 0.0, cfg.beta_max))
    reserved, residual, meta = allocate_hierarchical(
        cfg, state, rho, beta, use_shield=False,
        reserved_override=reserved, safety_info_override=sinfo,
    )
    meta.update(refmeta)
    meta.update({"delta_rho": float(rho - rho_ref), "delta_beta": float(beta - beta_ref)})
    return reserved, residual, meta


def demand_aware_controls(cfg: SimConfig, state: Dict[str, np.ndarray]) -> Tuple[float, float]:
    q = np.asarray(state["q_bytes"], dtype=np.float64)
    cqi = np.asarray(state["cqi"], dtype=np.int64)
    hol = np.asarray(state["hol_ms"], dtype=np.float64)
    deadlines = np.asarray(state["deadlines_ms"], dtype=np.float64)
    avg_rate = np.asarray(state["avg_rate_mbps"], dtype=np.float64)
    min_rates = np.asarray(state["min_rates_mbps"], dtype=np.float64)

    demand = 0.0
    urgencies = []
    for ue in cfg.urllc_ues:
        if q[ue] <= 0:
            continue
        demand += q[ue] / _bytes_per_prb(cfg, int(cqi[ue]))
        urgencies.append(float(np.clip(hol[ue] / max(deadlines[ue], 1e-6), 0.0, 2.0)))
    demand_ratio = np.clip(demand / max(1.0, cfg.n_prb), 0.0, 1.5)
    urgency = float(np.mean(urgencies)) if urgencies else 0.0
    rho = float(np.clip(0.12 + 0.55 * demand_ratio + 0.22 * urgency, cfg.rho_min, cfg.rho_max))

    deficits = []
    for ue in cfg.embb_ues:
        target = max(1e-6, float(min_rates[ue]))
        deficits.append(max(0.0, target - float(avg_rate[ue])) / target)
    mean_def = float(np.mean(deficits)) if deficits else 0.0
    beta = float(np.clip(1.0 + 5.0 * mean_def, 0.0, cfg.beta_max))
    return rho, beta
