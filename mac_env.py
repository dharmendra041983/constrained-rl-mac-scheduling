from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

# 15-level CQI spectral-efficiency abstraction (CQI 1..15).
CQI_EFF = np.array([
    0.0,
    0.1523, 0.2344, 0.3770, 0.6016, 0.8770,
    1.1758, 1.4766, 1.9141, 2.4063, 2.7305,
    3.3223, 3.9023, 4.5234, 5.1152, 5.5547,
], dtype=np.float64)


@dataclass
class SimConfig:
    # V2.6 retains the heterogeneous 10-UE cell from V2.3/V2.5.
    n_embb: int = 6
    n_urllc: int = 4
    n_prb: int = 106
    slot_ms: float = 1.0
    episode_steps: int = 6000

    embb_pkt_bytes: int = 1500
    embb_base_rates_mbps: Tuple[float, ...] = (3.2, 3.0, 2.8, 2.2, 2.0, 1.8)
    embb_min_rates_mbps: Tuple[float, ...] = (2.64, 2.40, 2.16, 1.44, 1.20, 1.08)

    urllc_pkt_bytes: Tuple[int, ...] = (256, 300, 384, 512)
    urllc_base_iat_ms: Tuple[float, ...] = (2.6, 2.2, 2.8, 3.2)
    urllc_deadlines_ms: Tuple[float, ...] = (4.0, 5.0, 6.0, 8.0)
    urllc_burst_rate_multiplier: float = 3.0
    urllc_burst_start_prob: float = 0.010
    urllc_burst_end_prob: float = 0.12

    embb_buffer_bytes: int = 400_000
    urllc_buffer_bytes: int = 120_000
    rate_warmup_steps: int = 400
    ewma_alpha: float = 0.02

    # Explicit long-term CMDP limits.
    epsilon_lat: float = 0.01
    epsilon_rate: float = 0.05
    epsilon_drop: float = 0.03

    cqi_min: int = 1
    cqi_max: int = 15
    cqi_update_steps: int = 5
    cqi_step_max: int = 1
    # 0..5 are eMBB; 6..9 are URLLC. Values deliberately span center/edge users.
    cqi_anchors: Tuple[int, ...] = (12, 11, 10, 8, 7, 6, 10, 8, 7, 6)
    mobile_ues: Tuple[int, ...] = (3, 5, 8, 9)
    mobility_update_steps: int = 80
    mobility_offset_max: int = 2

    # Approximate usable REs in one PRB during a 1-ms scheduling interval.
    re_per_prb: int = 168
    phy_overhead_factor: float = 0.86

    throughput_norm_mbps: float = 100.0

    # V2.5 minimal emergency shield. Unlike V2.4, the shield is not the main
    # URLLC scheduler; it protects only packets with very small remaining slack.
    shield_slack_ms: float = 1.0
    shield_burst_slack_ms: float = 1.5
    shield_packets_per_ue: int = 1
    shield_max_prb_fraction: float = 0.35

    # Hierarchical class-budget controller bounds. V2.6 anchors the learned
    # controls to a deadline-aware state-dependent reference and lets PPO learn
    # only bounded residual corrections around that reference.
    rho_min: float = 0.08
    rho_max: float = 0.82
    beta_max: float = 6.0
    static_rho: float = 0.34
    static_beta: float = 2.0
    reference_beta_base: float = 1.0
    delta_rho_max: float = 0.18
    delta_beta_max: float = 1.50

    @property
    def n_ue(self) -> int:
        return self.n_embb + self.n_urllc

    @property
    def embb_ues(self) -> List[int]:
        return list(range(self.n_embb))

    @property
    def urllc_ues(self) -> List[int]:
        return list(range(self.n_embb, self.n_ue))

    @property
    def edge_ues(self) -> List[int]:
        anchors = np.asarray(self.cqi_anchors)
        return np.where(anchors <= 8)[0].astype(int).tolist()

    @property
    def edge_embb_ues(self) -> List[int]:
        return [u for u in self.embb_ues if u in self.edge_ues]

    @property
    def center_embb_ues(self) -> List[int]:
        return [u for u in self.embb_ues if u not in self.edge_ues]

    @property
    def buffer_capacities(self) -> np.ndarray:
        return np.array(
            [self.embb_buffer_bytes] * self.n_embb + [self.urllc_buffer_bytes] * self.n_urllc,
            dtype=np.float64,
        )

    @property
    def min_rates(self) -> np.ndarray:
        x = np.zeros(self.n_ue, dtype=np.float64)
        x[: self.n_embb] = np.asarray(self.embb_min_rates_mbps, dtype=np.float64)
        return x

    @property
    def base_deadlines(self) -> np.ndarray:
        x = np.zeros(self.n_ue, dtype=np.float64)
        x[self.n_embb :] = np.asarray(self.urllc_deadlines_ms, dtype=np.float64)
        return x

    @property
    def cost_limits(self) -> np.ndarray:
        return np.array([self.epsilon_lat, self.epsilon_rate, self.epsilon_drop], dtype=np.float32)

    def validate(self) -> None:
        if len(self.embb_base_rates_mbps) != self.n_embb:
            raise ValueError("embb_base_rates_mbps length must equal n_embb")
        if len(self.embb_min_rates_mbps) != self.n_embb:
            raise ValueError("embb_min_rates_mbps length must equal n_embb")
        if len(self.urllc_pkt_bytes) != self.n_urllc:
            raise ValueError("urllc_pkt_bytes length must equal n_urllc")
        if len(self.urllc_base_iat_ms) != self.n_urllc:
            raise ValueError("urllc_base_iat_ms length must equal n_urllc")
        if len(self.urllc_deadlines_ms) != self.n_urllc:
            raise ValueError("urllc_deadlines_ms length must equal n_urllc")
        if len(self.cqi_anchors) != self.n_ue:
            raise ValueError("cqi_anchors length must equal n_ue")


@dataclass(frozen=True)
class Scenario:
    name: str
    # Static scales.
    load_scale: float = 1.0
    urllc_scale: float = 1.0
    global_cqi_shift: int = 0
    deadline_scale: float = 1.0

    # Dynamic eMBB load surge.
    surge_scale: float = 1.0
    surge_start: float = 2.0  # >1 disables
    surge_end: float = 2.0

    # Dynamic URLLC burst-intensity increase.
    burst_scale: float = 1.0
    burst_start: float = 2.0
    burst_end: float = 2.0

    # UE-specific channel fade.
    fade_ues: Tuple[int, ...] = ()
    fade_shift: int = 0
    fade_start: float = 2.0
    fade_end: float = 2.0

    # UE-specific traffic hotspot.
    hotspot_ues: Tuple[int, ...] = ()
    hotspot_scale: float = 1.0
    hotspot_start: float = 2.0
    hotspot_end: float = 2.0


SCENARIOS: Dict[str, Scenario] = {
    "nominal_hetero": Scenario("nominal_hetero"),
    "load_surge": Scenario(
        "load_surge", surge_scale=1.45, surge_start=0.35, surge_end=0.78
    ),
    "edge_fade": Scenario(
        "edge_fade", fade_ues=(4, 5, 8, 9), fade_shift=-3, fade_start=0.35, fade_end=1.0
    ),
    "bursty_urllc": Scenario(
        "bursty_urllc", burst_scale=2.0, burst_start=0.30, burst_end=0.78
    ),
    "mixed_dynamic": Scenario(
        "mixed_dynamic",
        surge_scale=1.30, surge_start=0.25, surge_end=0.78,
        burst_scale=1.65, burst_start=0.48, burst_end=0.88,
        fade_ues=(4, 5, 8, 9), fade_shift=-2, fade_start=0.42, fade_end=1.0,
    ),
    # Held-out heterogeneous hotspot: one edge eMBB and one edge URLLC become busy
    # while their channels degrade.
    "hotspot_unseen": Scenario(
        "hotspot_unseen",
        hotspot_ues=(2, 8), hotspot_scale=2.20, hotspot_start=0.38, hotspot_end=0.90,
        fade_ues=(2, 8), fade_shift=-3, fade_start=0.45, fade_end=0.95,
    ),
    "tight_sla": Scenario("tight_sla", deadline_scale=0.82),
}


class PacketQueue:
    """Finite byte buffer with packet arrival timestamps."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = int(capacity_bytes)
        self.packets: Deque[List[int]] = deque()  # [arrival_step, remaining_bytes]
        self.bytes_in_queue = 0

    def push(self, arrival_step: int, size_bytes: int) -> int:
        size_bytes = int(size_bytes)
        free = self.capacity_bytes - self.bytes_in_queue
        admitted = min(size_bytes, max(0, free))
        dropped = size_bytes - admitted
        if admitted > 0:
            self.packets.append([int(arrival_step), admitted])
            self.bytes_in_queue += admitted
        return dropped

    def hol_delay_ms(self, now_step: int, slot_ms: float) -> float:
        if not self.packets:
            return 0.0
        return max(0, now_step - self.packets[0][0]) * slot_ms

    def pop_bytes(self, bytes_to_send: int, now_step: int, slot_ms: float) -> Tuple[int, List[float]]:
        remaining = int(max(0, bytes_to_send))
        sent = 0
        completed_delays: List[float] = []
        while self.packets and remaining > 0:
            pkt = self.packets[0]
            take = min(pkt[1], remaining)
            pkt[1] -= take
            remaining -= take
            sent += take
            self.bytes_in_queue -= take
            if pkt[1] == 0:
                arrival_step = pkt[0]
                self.packets.popleft()
                completed_delays.append((now_step - arrival_step + 1) * slot_ms)
        return sent, completed_delays


class MacSchedulingEnv:
    """Heterogeneous packet/queue-level single-cell MAC abstraction.

    V2.6 retains heterogeneous service targets and dynamic per-UE conditions.
    The environment executes integer PRB allocations produced by a reference-
    anchored hierarchical class-budget controller or comparison schedulers.

    Cost vector order:
      [HOL deadline violation, eMBB minimum-rate violation, packet-drop ratio].
    """

    def __init__(self, cfg: SimConfig, seed: int = 0, scenario: Scenario | str = "nominal_hetero"):
        cfg.validate()
        self.cfg = cfg
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.obs_dim_per_ue = 8
        self.global_obs_dim = 10
        self.obs_dim = self.cfg.n_ue * self.obs_dim_per_ue + self.global_obs_dim
        self.reset(seed)

    def _frac(self) -> float:
        return min(1.0, self.t / max(1, self.cfg.episode_steps - 1))

    @staticmethod
    def _active(frac: float, start: float, end: float) -> bool:
        return start <= frac <= end

    def current_deadlines(self) -> np.ndarray:
        d = self.cfg.base_deadlines.copy()
        d[self.cfg.urllc_ues] *= self.scenario.deadline_scale
        return d

    def _is_hotspot(self, ue: int) -> bool:
        f = self._frac()
        return ue in self.scenario.hotspot_ues and self._active(
            f, self.scenario.hotspot_start, self.scenario.hotspot_end
        )

    def _embb_load_scale(self, ue: int) -> float:
        f = self._frac()
        s = self.scenario.load_scale
        if self._active(f, self.scenario.surge_start, self.scenario.surge_end):
            s *= self.scenario.surge_scale
        if self._is_hotspot(ue):
            s *= self.scenario.hotspot_scale
        return s

    def _urllc_load_scale(self, ue: int) -> float:
        f = self._frac()
        s = self.scenario.urllc_scale
        if self._active(f, self.scenario.burst_start, self.scenario.burst_end):
            s *= self.scenario.burst_scale
        if self._is_hotspot(ue):
            s *= self.scenario.hotspot_scale
        return s

    def _cqi_shift(self, ue: int) -> int:
        f = self._frac()
        s = int(self.scenario.global_cqi_shift)
        if ue in self.scenario.fade_ues and self._active(f, self.scenario.fade_start, self.scenario.fade_end):
            s += int(self.scenario.fade_shift)
        return s

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        caps = self.cfg.buffer_capacities
        self.queues = [PacketQueue(int(caps[i])) for i in range(self.cfg.n_ue)]

        anchors = np.asarray(self.cfg.cqi_anchors, dtype=np.int64)
        jitter = self.rng.integers(-1, 2, size=self.cfg.n_ue)
        self.base_cqi = np.clip(anchors + jitter, self.cfg.cqi_min, self.cfg.cqi_max).astype(np.int64)
        self.mobility_offset = np.zeros(self.cfg.n_ue, dtype=np.int64)
        self._refresh_observed_cqi()

        self.avg_rate_mbps = np.zeros(self.cfg.n_ue, dtype=np.float64)
        self.avg_rate_mbps[: self.cfg.n_embb] = np.asarray(self.cfg.embb_min_rates_mbps, dtype=np.float64)
        self.prev_prbs = np.zeros(self.cfg.n_ue, dtype=np.int64)
        self.urllc_burst_on = np.zeros(self.cfg.n_urllc, dtype=bool)

        self.total_arrived_bytes = 0
        self.total_dropped_bytes = 0
        self.total_arrived_by_ue = np.zeros(self.cfg.n_ue, dtype=np.float64)
        self.total_dropped_by_ue = np.zeros(self.cfg.n_ue, dtype=np.float64)
        self.total_sent_bytes = np.zeros(self.cfg.n_ue, dtype=np.float64)
        self.urllc_completed_delays: List[List[float]] = [[] for _ in range(self.cfg.n_urllc)]
        self.cost_history: List[np.ndarray] = []
        self.rate_violation_history: List[float] = []
        self.prb_history: List[np.ndarray] = []
        self.reserved_prb_history: List[np.ndarray] = []
        self.safety_active_history: List[float] = []
        self.safety_cap_hit_history: List[float] = []
        self.throughput_history_mbps: List[float] = []

        # Current-slot arrivals are visible before the first scheduling decision.
        self._generate_arrivals()
        return self.observe()

    def _refresh_observed_cqi(self) -> None:
        cqi = self.base_cqi.astype(np.int64) + self.mobility_offset
        for ue in range(self.cfg.n_ue):
            cqi[ue] += self._cqi_shift(ue)
        self.cqi = np.clip(cqi, self.cfg.cqi_min, self.cfg.cqi_max).astype(np.int64)

    def _update_channel(self) -> None:
        if self.t > 0 and self.t % self.cfg.cqi_update_steps == 0:
            anchors = np.asarray(self.cfg.cqi_anchors, dtype=np.int64)
            noise = self.rng.integers(-self.cfg.cqi_step_max, self.cfg.cqi_step_max + 1, size=self.cfg.n_ue)
            # Weak mean reversion keeps center/edge identity while allowing fading.
            reversion = np.sign(anchors - self.base_cqi).astype(np.int64)
            take_reversion = self.rng.random(self.cfg.n_ue) < 0.35
            delta = noise + reversion * take_reversion.astype(np.int64)
            self.base_cqi = np.clip(self.base_cqi + delta, self.cfg.cqi_min, self.cfg.cqi_max)

        if self.t > 0 and self.t % self.cfg.mobility_update_steps == 0:
            for ue in self.cfg.mobile_ues:
                step = int(self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]))
                self.mobility_offset[ue] = int(np.clip(
                    self.mobility_offset[ue] + step,
                    -self.cfg.mobility_offset_max,
                    self.cfg.mobility_offset_max,
                ))
        self._refresh_observed_cqi()

    def _generate_arrivals(self) -> Tuple[int, int]:
        arrived = 0
        dropped = 0
        slot_seconds = self.cfg.slot_ms / 1000.0

        # Heterogeneous eMBB offered load.
        for local_idx, ue in enumerate(self.cfg.embb_ues):
            rate = float(self.cfg.embb_base_rates_mbps[local_idx]) * self._embb_load_scale(ue)
            mean_bytes = rate * 1e6 / 8.0 * slot_seconds
            n_bytes = int(self.rng.poisson(mean_bytes))
            arrived += n_bytes
            self.total_arrived_by_ue[ue] += n_bytes
            left = n_bytes
            while left > 0:
                pkt = min(self.cfg.embb_pkt_bytes, left)
                d = self.queues[ue].push(self.t, pkt)
                dropped += d
                self.total_dropped_by_ue[ue] += d
                left -= pkt

        # Markov-modulated bursty URLLC. More than one packet can arrive in a slot.
        for local_idx, ue in enumerate(self.cfg.urllc_ues):
            load_scale = self._urllc_load_scale(ue)
            if self.urllc_burst_on[local_idx]:
                p_end = min(0.8, self.cfg.urllc_burst_end_prob / max(0.75, load_scale ** 0.25))
                if self.rng.random() < p_end:
                    self.urllc_burst_on[local_idx] = False
            else:
                p_start = min(0.25, self.cfg.urllc_burst_start_prob * load_scale)
                if self.rng.random() < p_start:
                    self.urllc_burst_on[local_idx] = True

            burst_mult = self.cfg.urllc_burst_rate_multiplier if self.urllc_burst_on[local_idx] else 1.0
            mean_iat = float(self.cfg.urllc_base_iat_ms[local_idx]) / max(0.1, load_scale * burst_mult)
            lam = self.cfg.slot_ms / max(0.1, mean_iat)
            n_pkts = int(self.rng.poisson(lam))
            pkt_size = int(self.cfg.urllc_pkt_bytes[local_idx])
            bytes_in = n_pkts * pkt_size
            arrived += bytes_in
            self.total_arrived_by_ue[ue] += bytes_in
            for _ in range(n_pkts):
                d = self.queues[ue].push(self.t, pkt_size)
                dropped += d
                self.total_dropped_by_ue[ue] += d

        self.total_arrived_bytes += arrived
        self.total_dropped_bytes += dropped
        return arrived, dropped

    def _prb_capacity_bytes(self, cqi: int, prbs: int) -> int:
        eff = CQI_EFF[int(cqi)]
        bits = eff * int(prbs) * self.cfg.re_per_prb * self.cfg.phy_overhead_factor
        return int(max(0.0, bits / 8.0))

    @staticmethod
    def _largest_remainder(weights: np.ndarray, n_prb: int) -> np.ndarray:
        raw = np.asarray(weights, dtype=np.float64) * int(n_prb)
        out = np.floor(raw).astype(np.int64)
        rem = int(n_prb - out.sum())
        if rem > 0:
            frac = raw - out
            out[np.argsort(-frac)[:rem]] += 1
        return out

    def project_action(self, logits: np.ndarray, reserved_prbs: np.ndarray | None = None) -> np.ndarray:
        """Project policy logits onto the residual PRB budget.

        ``reserved_prbs`` is an immutable action-space reservation computed
        before the learned action. PPO may redistribute only the remaining PRBs.
        """
        logits = np.asarray(logits, dtype=np.float64).reshape(self.cfg.n_ue)
        nonempty = np.array([q.bytes_in_queue > 0 for q in self.queues], dtype=bool)
        if reserved_prbs is None:
            reserved = np.zeros(self.cfg.n_ue, dtype=np.int64)
        else:
            reserved = np.maximum(0, np.asarray(reserved_prbs, dtype=np.int64).reshape(self.cfg.n_ue))
            reserved[~nonempty] = 0
        if int(reserved.sum()) > self.cfg.n_prb:
            raise ValueError("reserved PRBs exceed total PRB budget")
        available = int(self.cfg.n_prb - reserved.sum())
        if not np.any(nonempty) or available <= 0:
            return reserved.copy()

        masked = logits.copy()
        masked[~nonempty] = -1e9
        m = np.max(masked[nonempty])
        ex = np.zeros_like(masked)
        ex[nonempty] = np.exp(np.clip(masked[nonempty] - m, -50.0, 50.0))
        weights = ex / max(ex.sum(), 1e-12)
        residual = self._largest_remainder(weights, available)
        residual[~nonempty] = 0
        diff = available - int(residual.sum())
        if diff > 0:
            best = int(np.argmax(np.where(nonempty, weights, -1.0)))
            residual[best] += diff
        return reserved + residual

    def observe(self) -> np.ndarray:
        q = np.array([x.bytes_in_queue for x in self.queues], dtype=np.float64)
        hol = np.array([x.hol_delay_ms(self.t, self.cfg.slot_ms) for x in self.queues], dtype=np.float64)
        deadlines = self.current_deadlines()
        min_rates = self.cfg.min_rates
        caps = self.cfg.buffer_capacities

        cqi_n = (self.cqi - self.cfg.cqi_min) / max(1, self.cfg.cqi_max - self.cfg.cqi_min)
        q_n = np.clip(q / np.maximum(1.0, caps), 0.0, 1.0)
        urgency = np.zeros(self.cfg.n_ue, dtype=np.float64)
        uidx = np.array(self.cfg.urllc_ues, dtype=int)
        urgency[uidx] = np.clip(hol[uidx] / np.maximum(deadlines[uidx], 1e-6), 0.0, 2.5)

        rate_ratio = np.zeros(self.cfg.n_ue, dtype=np.float64)
        eidx = np.array(self.cfg.embb_ues, dtype=int)
        rate_ratio[eidx] = np.clip(
            self.avg_rate_mbps[eidx] / np.maximum(min_rates[eidx], 1e-6), 0.0, 2.5
        )
        prev_n = np.clip(self.prev_prbs / max(1, self.cfg.n_prb), 0.0, 1.0)
        cls = np.zeros(self.cfg.n_ue, dtype=np.float64)
        cls[uidx] = 1.0
        sla_target = np.zeros(self.cfg.n_ue, dtype=np.float64)
        sla_target[eidx] = min_rates[eidx] / max(1e-6, max(self.cfg.embb_min_rates_mbps))
        sla_target[uidx] = deadlines[uidx] / max(1e-6, max(self.cfg.urllc_deadlines_ms))
        edge = np.zeros(self.cfg.n_ue, dtype=np.float64)
        edge[self.cfg.edge_ues] = 1.0

        per_ue = np.stack(
            [cqi_n, q_n, urgency, rate_ratio, prev_n, cls, sla_target, edge], axis=1
        ).reshape(-1)

        # V2.6 global constraint-pressure features make the high-level control
        # problem explicit instead of forcing the actor to infer these aggregates
        # from 80 individual UE features. All values are scaled to O(1).
        q_urllc = float(np.sum(q[uidx]))
        q_total = float(np.sum(q))
        demand_prbs = 0.0
        active_urgency = []
        for ue in self.cfg.urllc_ues:
            if q[ue] <= 0:
                continue
            eff = float(CQI_EFF[int(self.cqi[ue])])
            bpp = max(1.0, eff * self.cfg.re_per_prb * self.cfg.phy_overhead_factor / 8.0)
            demand_prbs += q[ue] / bpp
            active_urgency.append(float(np.clip(hol[ue] / max(deadlines[ue], 1e-6), 0.0, 2.5)))
        urg = np.asarray(active_urgency, dtype=np.float64) if active_urgency else np.zeros(1)
        deficits = np.maximum(0.0, min_rates[eidx] - self.avg_rate_mbps[eidx]) / np.maximum(min_rates[eidx], 1e-6)
        urllc_cqi_n = cqi_n[uidx] if len(uidx) else np.zeros(1)
        pressure = np.array([
            min(2.0, demand_prbs / max(1.0, self.cfg.n_prb)) / 2.0,
            min(2.0, float(np.mean(urg))) / 2.0,
            min(2.0, float(np.max(urg))) / 2.0,
            float(np.mean(urg >= 0.50)),
            float(np.mean(urg >= 0.80)),
            float(np.clip(np.mean(deficits), 0.0, 1.5)) / 1.5,
            float(np.clip(np.max(deficits), 0.0, 1.5)) / 1.5,
            q_urllc / max(1.0, q_total),
            float(np.mean(self.urllc_burst_on)) if self.cfg.n_urllc else 0.0,
            1.0 - float(np.mean(urllc_cqi_n)),
        ], dtype=np.float64)
        obs = np.concatenate([per_ue, pressure])
        return obs.astype(np.float32)

    def raw_state(self) -> Dict[str, np.ndarray]:
        return {
            "cqi": self.cqi.copy(),
            "q_bytes": np.array([q.bytes_in_queue for q in self.queues], dtype=np.float64),
            "hol_ms": np.array([q.hol_delay_ms(self.t, self.cfg.slot_ms) for q in self.queues], dtype=np.float64),
            "avg_rate_mbps": self.avg_rate_mbps.copy(),
            "prev_prbs": self.prev_prbs.copy(),
            "deadlines_ms": self.current_deadlines().copy(),
            "min_rates_mbps": self.cfg.min_rates.copy(),
            "edge_mask": np.array([u in self.cfg.edge_ues for u in range(self.cfg.n_ue)], dtype=bool),
            "burst_on": np.concatenate([
                np.zeros(self.cfg.n_embb, dtype=bool), self.urllc_burst_on.copy()
            ]),
        }

    def step(self, action_logits: np.ndarray, reserved_prbs: np.ndarray | None = None, safety_info: Dict[str, float] | None = None):
        """Execute one decision on the exact state that was observed."""
        pre_hol = np.array(
            [q.hol_delay_ms(self.t, self.cfg.slot_ms) for q in self.queues], dtype=np.float64
        )
        deadlines = self.current_deadlines()

        prbs = self.project_action(action_logits, reserved_prbs=reserved_prbs)
        self.prev_prbs = prbs.copy()

        sent = np.zeros(self.cfg.n_ue, dtype=np.float64)
        for ue in range(self.cfg.n_ue):
            capacity = self._prb_capacity_bytes(self.cqi[ue], int(prbs[ue]))
            sent_bytes, completed_delays = self.queues[ue].pop_bytes(capacity, self.t, self.cfg.slot_ms)
            sent[ue] = sent_bytes
            if ue in self.cfg.urllc_ues:
                local = ue - self.cfg.n_embb
                self.urllc_completed_delays[local].extend(completed_delays)

        slot_seconds = self.cfg.slot_ms / 1000.0
        inst_rate_mbps = sent * 8.0 / 1e6 / slot_seconds
        a = self.cfg.ewma_alpha
        self.avg_rate_mbps = (1.0 - a) * self.avg_rate_mbps + a * inst_rate_mbps
        self.total_sent_bytes += sent
        cell_thr_mbps = float(sent.sum() * 8.0 / 1e6 / slot_seconds)
        reward = cell_thr_mbps / self.cfg.throughput_norm_mbps

        self.t += 1
        done = self.t >= self.cfg.episode_steps
        if not done:
            self._update_channel()
            arrived_bytes, dropped_bytes = self._generate_arrivals()
        else:
            arrived_bytes, dropped_bytes = 0, 0

        post_hol = np.array(
            [q.hol_delay_ms(self.t, self.cfg.slot_ms) for q in self.queues], dtype=np.float64
        )
        if self.cfg.n_urllc:
            uidx = np.array(self.cfg.urllc_ues, dtype=int)
            # Current deadlines can change only by scenario-level scaling, not per slot.
            lat_cost = float(np.mean(post_hol[uidx] > deadlines[uidx]))
        else:
            lat_cost = 0.0

        if self.t >= self.cfg.rate_warmup_steps and self.cfg.n_embb:
            elapsed_s = max(1, self.t) * slot_seconds
            cumulative_rate_mbps = self.total_sent_bytes * 8.0 / 1e6 / elapsed_s
            eidx = np.array(self.cfg.embb_ues, dtype=int)
            targets = np.asarray(self.cfg.embb_min_rates_mbps, dtype=np.float64)
            rates = cumulative_rate_mbps[eidx]
            normalized_deficit = np.maximum(0.0, targets - rates) / np.maximum(targets, 1e-6)
            rate_cost = float(np.mean(normalized_deficit))
            rate_violation_fraction = float(np.mean(rates < targets))
        else:
            rate_cost = 0.0
            rate_violation_fraction = 0.0

        drop_cost = float(dropped_bytes / arrived_bytes) if arrived_bytes > 0 else 0.0
        costs = np.array([lat_cost, rate_cost, drop_cost], dtype=np.float32)

        x = self.avg_rate_mbps
        fairness = float((x.sum() ** 2) / (self.cfg.n_ue * np.square(x).sum() + 1e-12)) if np.any(x > 0) else 1.0
        self.cost_history.append(costs.copy())
        self.rate_violation_history.append(rate_violation_fraction)
        self.prb_history.append(prbs.copy())
        reserved_arr = np.zeros(self.cfg.n_ue, dtype=np.int64) if reserved_prbs is None else np.asarray(reserved_prbs, dtype=np.int64).copy()
        self.reserved_prb_history.append(reserved_arr)
        sinfo = safety_info or {}
        self.safety_active_history.append(float(sinfo.get("safety_active", reserved_arr.sum() > 0)))
        self.safety_cap_hit_history.append(float(sinfo.get("safety_cap_hit", 0.0)))
        self.throughput_history_mbps.append(cell_thr_mbps)

        obs = self.observe()
        info = {
            "costs": costs,
            "throughput_mbps": cell_thr_mbps,
            "embb_throughput_mbps": float(inst_rate_mbps[self.cfg.embb_ues].sum()),
            "urllc_throughput_mbps": float(inst_rate_mbps[self.cfg.urllc_ues].sum()),
            "fairness": fairness,
            "prbs": prbs.copy(),
            "cqi": self.cqi.copy(),
            "pre_hol_ms": pre_hol,
            "post_hol_ms": post_hol,
            "deadlines_ms": deadlines.copy(),
            "arrived_bytes": arrived_bytes,
            "dropped_bytes": dropped_bytes,
            "sent_bytes": sent.copy(),
            "rate_deficit_cost": rate_cost,
            "rate_violation_fraction": rate_violation_fraction,
            "reserved_prbs": np.zeros(self.cfg.n_ue, dtype=np.int64) if reserved_prbs is None else np.asarray(reserved_prbs, dtype=np.int64).copy(),
        }
        return obs, float(reward), costs, done, info

    def metrics(self) -> Dict[str, float]:
        costs = np.vstack(self.cost_history) if self.cost_history else np.zeros((1, 3))
        prbs = np.vstack(self.prb_history) if self.prb_history else np.zeros((1, self.cfg.n_ue))
        reserved = np.vstack(self.reserved_prb_history) if self.reserved_prb_history else np.zeros((1, self.cfg.n_ue))
        slot_seconds = self.cfg.slot_ms / 1000.0
        duration_s = max(1, self.t) * slot_seconds
        ue_thr = self.total_sent_bytes * 8.0 / 1e6 / duration_s
        eidx = np.array(self.cfg.embb_ues, dtype=int)
        uidx = np.array(self.cfg.urllc_ues, dtype=int)
        min_rates = np.asarray(self.cfg.embb_min_rates_mbps, dtype=np.float64)
        deadlines = self.current_deadlines()[uidx]

        all_delays = np.concatenate([
            np.asarray(x, dtype=np.float64) for x in self.urllc_completed_delays if len(x) > 0
        ]) if any(len(x) > 0 for x in self.urllc_completed_delays) else np.array([], dtype=np.float64)
        if all_delays.size:
            p50, p95, p99 = np.percentile(all_delays, [50, 95, 99])
        else:
            p50 = p95 = p99 = 0.0

        miss_rates = []
        completed_counts = []
        late_counts = []
        for k, arr_list in enumerate(self.urllc_completed_delays):
            arr = np.asarray(arr_list, dtype=np.float64)
            completed_counts.append(int(arr.size))
            late = int(np.sum(arr > deadlines[k])) if arr.size else 0
            late_counts.append(late)
            miss_rates.append(float(late / arr.size) if arr.size else 0.0)
        packet_deadline_miss = float(sum(late_counts) / max(1, sum(completed_counts)))
        worst_packet_miss = float(max(miss_rates)) if miss_rates else 0.0

        rate_ratios = ue_thr[eidx] / np.maximum(min_rates, 1e-9) if self.cfg.n_embb else np.array([1.0])
        fairness = float((ue_thr.sum() ** 2) / (self.cfg.n_ue * np.square(ue_thr).sum() + 1e-12)) if np.any(ue_thr > 0) else 1.0

        center = np.array(self.cfg.center_embb_ues, dtype=int)
        edge = np.array(self.cfg.edge_embb_ues, dtype=int)
        return {
            "throughput_mbps": float(np.mean(self.throughput_history_mbps)) if self.throughput_history_mbps else 0.0,
            "embb_throughput_mbps": float(ue_thr[eidx].sum()) if self.cfg.n_embb else 0.0,
            "urllc_throughput_mbps": float(ue_thr[uidx].sum()) if self.cfg.n_urllc else 0.0,
            "center_embb_throughput_mbps": float(ue_thr[center].sum()) if center.size else 0.0,
            "edge_embb_throughput_mbps": float(ue_thr[edge].sum()) if edge.size else 0.0,
            "min_embb_throughput_mbps": float(np.min(ue_thr[eidx])) if self.cfg.n_embb else 0.0,
            "min_embb_satisfaction_ratio": float(np.min(rate_ratios)),
            "mean_embb_satisfaction_ratio": float(np.mean(rate_ratios)),
            "latency_violation": float(costs[:, 0].mean()),
            "rate_deficit_cost": float(costs[:, 1].mean()),
            "rate_violation_fraction": float(np.mean(self.rate_violation_history)) if self.rate_violation_history else 0.0,
            # Backward-compatible alias; V2.5 paper text should use rate_deficit_cost.
            "rate_violation": float(np.mean(self.rate_violation_history)) if self.rate_violation_history else 0.0,
            "drop_ratio": float(self.total_dropped_bytes / max(1, self.total_arrived_bytes)),
            "packet_deadline_miss_ratio": packet_deadline_miss,
            "worst_urllc_deadline_miss_ratio": worst_packet_miss,
            "urllc_delay_p50_ms": float(p50),
            "urllc_delay_p95_ms": float(p95),
            "urllc_delay_p99_ms": float(p99),
            "jain_fairness": fairness,
            "avg_prb_embb": float(prbs[:, eidx].sum(axis=1).mean()) if self.cfg.n_embb else 0.0,
            "avg_prb_urllc": float(prbs[:, uidx].sum(axis=1).mean()) if self.cfg.n_urllc else 0.0,
            "avg_safety_reserved_prbs": float(reserved.sum(axis=1).mean()),
            "safety_activation_fraction": float(np.mean(self.safety_active_history)) if self.safety_active_history else 0.0,
            "safety_cap_hit_fraction": float(np.mean(self.safety_cap_hit_history)) if self.safety_cap_hit_history else 0.0,
            "observed_urllc_packets": int(sum(completed_counts)),
        }
