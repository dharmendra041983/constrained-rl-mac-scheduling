from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from mac_env import CQI_EFF, SimConfig


def emergency_shield(cfg: SimConfig, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Reserve a small immutable PRB budget only for imminently expiring URLLC packets.

    V2.5 deliberately avoids using the shield as the primary URLLC scheduler.
    It activates only when remaining HOL slack is very small. The hierarchical
    controller remains responsible for proactive class-level resource budgeting.
    """
    reserved = np.zeros(cfg.n_ue, dtype=np.int64)
    q = np.asarray(state["q_bytes"], dtype=np.float64)
    hol = np.asarray(state["hol_ms"], dtype=np.float64)
    deadlines = np.asarray(state["deadlines_ms"], dtype=np.float64)
    cqi = np.asarray(state["cqi"], dtype=np.int64)
    burst = np.asarray(state.get("burst_on", np.zeros(cfg.n_ue, dtype=bool)), dtype=bool)

    cap = int(np.floor(cfg.shield_max_prb_fraction * cfg.n_prb))
    candidates = []
    for local_idx, ue in enumerate(cfg.urllc_ues):
        if q[ue] <= 0:
            continue
        slack = float(deadlines[ue] - hol[ue])
        threshold = cfg.shield_burst_slack_ms if burst[ue] else cfg.shield_slack_ms
        if slack > threshold:
            continue
        urgency = float(hol[ue] / max(deadlines[ue], 1e-6))
        candidates.append((slack, -urgency, ue, local_idx))

    candidates.sort()  # smallest slack first
    cap_hit = False
    for _, _, ue, local_idx in candidates:
        eff = float(CQI_EFF[int(cqi[ue])])
        bytes_per_prb = max(1.0, eff * cfg.re_per_prb * cfg.phy_overhead_factor / 8.0)
        pkt_bytes = int(cfg.urllc_pkt_bytes[local_idx])
        target_bytes = min(float(q[ue]), float(cfg.shield_packets_per_ue * pkt_bytes))
        need = max(1, int(np.ceil(target_bytes / bytes_per_prb)))
        remaining = cap - int(reserved.sum())
        if remaining <= 0:
            cap_hit = True
            break
        give = min(need, remaining)
        reserved[ue] += give
        if give < need:
            cap_hit = True
            break

    return reserved, {
        "safety_reserved_prbs": float(reserved.sum()),
        "safety_active": float(reserved.sum() > 0),
        "safety_cap_hit": float(cap_hit),
        "safety_reserved_fraction": float(reserved.sum() / max(1, cfg.n_prb)),
    }
