"""Gymnasium environment: UAV multi-priority resource allocation.

A single UAV base station hovers at fixed altitude and serves 3 traffic
classes (Emergency, Important, Normal).  Each slot the agent chooses bandwidth
shares ``b_k`` and power shares ``p_k`` (both sum to 1 via softmax).  Per class:

    R_k = b_k * log2(1 + p_k * g_k / (N0 * b_k))         [bps]
    Q_k(t+1) = max(0, Q_k - R_k * dt) + A_k               [bits]
    D_k = Q_k / max(R_k, eps)                             [s], clipped at 1 s

Observation (17-dim, normalized): plan.md Section 4 says "16-dim" but lists
3+3+3+1+1+6 = 17 components; we follow the explicit list (decision logged in
docs/decision_log.md).
    log10(g_k) scaled (3), log1p(Q_k) scaled (3),
    D_k / D_th clipped to [0, 5] (3), battery fraction (1),
    fraction of Emergency violations in last W=50 slots (1),
    previous action (6).

Action: 6 values in [-1, 1]; first 3 -> bandwidth shares, last 3 -> power
shares via softmax(3 * x).

Reward:
    r = w_emg*r_emg + w_imp*r_imp + w_norm*r_norm
        - lambda_pen * 1[D_emg > D_th]
    r_emg = -clip(D_emg / D_th, 0, 5)^2
    r_imp = log(1 + min(R_imp / R_target, 1))
    r_norm = R_norm / R_norm_ref

Scope limits (logged in docs/decision_log.md D7): fixed hover, single UAV, no
inter-cell interference.  Battery is a simple drain model (obs feature only).

The per-slot hot path uses scalar math over the 3 classes so the bare
environment runs at >= 20,000 steps/s on a laptop CPU (plan.md rule 8).
"""

from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .channel import Channel, noise_power_w
from .scenarios import Scenario
from .traffic import Traffic

EPS = 1e-9


class UAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, scenario=None, seed=None):
        super().__init__()
        self.config = dict(config or {})
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # ---- timing ----
        t = self.config.get("timing", {})
        self.slot_ms = float(t.get("slot_ms", 5.0))
        self.dt = self.slot_ms / 1000.0
        self.episode_slots = int(t.get("episode_slots", 2000))
        self.D_th = float(t.get("D_th", 15.0)) / 1000.0  # ms -> seconds

        # ---- channel ----
        c = self.config.get("channel", {})
        self.channel = Channel(
            n_classes=3,
            H=float(c.get("H", 100.0)),
            fc=float(c.get("fc", 2.0e9)),
            a=float(c.get("a", 9.61)),
            b=float(c.get("b", 0.16)),
            eta_los_db=float(c.get("eta_LoS_db", 1.0)),
            eta_nlos_db=float(c.get("eta_NLoS_db", 20.0)),
            los_stay_prob=float(c.get("los_stay_prob", 0.98)),
            rician_K_db=float(c.get("rician_K_db", 10.0)),
            fading_ar1=float(c.get("fading_ar1", 0.9)),
            noise_dbm_hz=float(c.get("noise_dbm_hz", -174.0)),
            noise_figure_db=float(c.get("noise_figure_db", 7.0)),
            rng=self.rng,
        )
        self.B_max = float(c.get("B_max", 1.0e7))
        self.P_max = float(c.get("P_max", 1.0))
        self.n0 = noise_power_w(1.0, self.channel.noise_dbm_hz,
                                self.channel.noise_figure_db)  # W/Hz

        # ---- users (fixed ground radius per episode) ----
        radius = self.config.get("users", {}).get(
            "radius_m", [400.0, 250.0, 100.0]
        )  # Emergency farthest: greedy starves it, strict_priority protects it
        self.radius_m = np.array(radius, dtype=float)

        # ---- traffic ----
        tr = self.config.get("traffic", {})
        self.queue_cap_bits = float(tr.get("queue_cap_mbit", 2.0)) * 1e6
        self.delay_clip_s = float(tr.get("delay_clip_s", 1.0))
        self.log1p_cap = math.log1p(self.queue_cap_bits)
        base_means = [
            float(tr.get("emergency_mean_mbps", 0.5)),
            float(tr.get("important_mean_mbps", 8.0)),
            float(tr.get("normal_mean_mbps", 6.0)),
        ]
        self._base_mean_mbps = list(base_means)
        self.traffic = Traffic(
            mean_mbps_per_class=base_means,
            packet_bytes=int(tr.get("packet_bytes", 200)),
            rng=self.rng,
        )

        # ---- reward ----
        rw = self.config.get("reward", {})
        self.w_emg = float(rw.get("w_emg", 1.0))
        self.w_imp = float(rw.get("w_imp", 0.5))
        self.w_norm = float(rw.get("w_norm", 0.1))
        self.lambda_pen = float(rw.get("lambda_pen", 5.0))
        self.R_target = float(rw.get("R_target", 6.0e6))
        rnf = rw.get("R_norm_ref", None)
        self.R_norm_ref = float(rnf) if rnf is not None else None

        # ---- battery ----
        bt = self.config.get("battery", {})
        self.battery_capacity_J = float(bt.get("capacity_J", 100.0))
        self.battery_fraction = 1.0

        # ---- scenario ----
        self.scenario = scenario
        self.ood_flag = False

        # ---- spaces ----
        # NOTE: plan.md Section 4 says "16-dim" but lists 17 components;
        # we follow the explicit list (decision logged).
        self.n_obs = 17
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32
        )

        # ---- episode state ----
        self.slot = 0
        self.queue_bits = np.zeros(3)
        self.rates_bps = np.zeros(3)
        self.delays_s = np.zeros(3)
        self.drops_bits = np.zeros(3)
        self.prev_action = np.zeros(6)
        self._raw_action = np.zeros(6)
        self._last_gain = np.zeros(3)
        self.violation_history = []
        # Domain-randomization overrides (applied per training episode).
        self._dr_shadow_db = 0.0
        self._dr_surge_mult = 1.0

    def randomize_episode(self, rng, dr_config):
        """Apply domain randomization for one training episode.

        Perturbs per-class arrival rates by [lo, hi], adds uniform shadowing in
        [0, shadow_max_db], and applies an emergency surge multiplier in
        [1.0, surge_max].  Nominal (non-randomized) evaluation is unaffected.
        """
        lo = float(dr_config.get("arrival_rate_min", 0.8))
        hi = float(dr_config.get("arrival_rate_max", 1.2))
        mults = rng.uniform(lo, hi, 3)
        for i, src in enumerate(self.traffic.sources):
            base = self._base_mean_mbps[i]
            src.mean_mbps = base * float(mults[i])
            src.lambda_packets_per_s = (src.mean_mbps * 1e6) / (src.packet_bytes * 8.0)
        self._dr_shadow_db = float(rng.uniform(0.0, float(dr_config.get("shadow_max_db", 3.0))))
        self._dr_surge_mult = float(rng.uniform(1.0, float(dr_config.get("surge_max", 1.5))))

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)
            self.channel.rng = self.rng
            # Rebuild the traffic sources so they draw from the seeded RNG
            # (TrafficSource objects hold their own rng references).
            tr = self.config.get("traffic", {})
            self.traffic = Traffic(
                mean_mbps_per_class=[
                    float(tr.get("emergency_mean_mbps", 0.5)),
                    float(tr.get("important_mean_mbps", 8.0)),
                    float(tr.get("normal_mean_mbps", 6.0)),
                ],
                packet_bytes=int(tr.get("packet_bytes", 200)),
                rng=self.rng,
            )
            if self.scenario is not None:
                self.scenario.rng = self.rng
        # Seed the gym spaces so action sampling is deterministic given a seed.
        try:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        except Exception:
            pass

        self.slot = 0
        self.queue_bits = np.zeros(3)
        self.rates_bps = np.zeros(3)
        self.delays_s = np.zeros(3)
        self.drops_bits = np.zeros(3)
        self.prev_action = np.zeros(6)
        self._raw_action = np.zeros(6)
        self._last_gain = np.zeros(3)
        self.violation_history = []
        self.battery_fraction = 1.0
        if self.scenario is not None:
            self.scenario.active = False
            self.scenario.start_slot = None
            self.scenario.end_slot = None
            self.scenario._schedule()
        self.ood_flag = bool(self.scenario.ood) if self.scenario is not None else False

        # Precompute fixed per-class geometry, then re-draw channel state.
        self.channel.set_geometry(self.radius_m)
        p0 = self.channel.los_probability(self.radius_m)
        self.channel.los_state = self.rng.random(3) < p0
        self.channel.fading_state = self.channel._draw_fading(self.channel.los_state)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(6)
        raw = action.copy()

        # Map action -> shares via softmax(3 * x) (scalar over 6 elements).
        def _softmax3(v):
            m = max(v)
            e = [math.exp(xi - m) for xi in v]
            s = sum(e)
            return [xi / s for xi in e]

        b_share = _softmax3([3.0 * xi for xi in action[:3]])
        p_share = _softmax3([3.0 * xi for xi in action[3:]])
        b_hz = [s * self.B_max for s in b_share]
        p_w = [s * self.P_max for s in p_share]

        # Scenario for this slot.
        if self.scenario is not None:
            self.ood_flag = bool(self.scenario.ood) and self.scenario.is_active(self.slot)
            multipliers = list(self.scenario.multipliers([1.0, 1.0, 1.0]))
            shadow_db = self.scenario.shadow_db()
        else:
            multipliers = [1.0, 1.0, 1.0]
            shadow_db = 0.0
        # Apply domain-randomization overrides (training only).
        multipliers[0] *= self._dr_surge_mult
        shadow_db += self._dr_shadow_db

        # Channel gain for this slot (geometry precomputed at reset).
        gain = self.channel.update(shadow_db=shadow_db)
        self._last_gain = gain

        # Rates per class (scalar Shannon).
        n0 = self.n0
        rates = []
        for k in range(3):
            b = b_hz[k]
            p = p_w[k]
            g = gain[k]
            if b > 0.0 and p > 0.0 and g > 0.0:
                rates.append(b * math.log2(1.0 + p * g / (n0 * b)))
            else:
                rates.append(0.0)
        self.rates_bps = np.array(rates, dtype=float)

        # Arrivals.
        arrivals = self.traffic.arrivals_bits(self.dt, multipliers=multipliers)

        # Queue update + drops (cap per class), scalar over the 3 classes.
        served = self.rates_bps * self.dt
        new_queue = [
            max(0.0, float(self.queue_bits[k]) - float(served[k])) + float(arrivals[k])
            for k in range(3)
        ]
        overflow = [
            max(0.0, new_queue[k] - self.queue_cap_bits) for k in range(3)
        ]
        self.drops_bits = np.array(overflow, dtype=float)
        self.queue_bits = np.array(
            [min(new_queue[k], self.queue_cap_bits) for k in range(3)],
            dtype=float,
        )

        # Delays.
        delays = [
            min(float(self.queue_bits[k]) / max(float(self.rates_bps[k]), EPS), self.delay_clip_s)
            for k in range(3)
        ]
        self.delays_s = np.array(delays, dtype=float)

        # Battery drain.
        p_used_W = p_w[0] + p_w[1] + p_w[2]
        self.battery_fraction = max(
            0.0,
            self.battery_fraction - p_used_W * self.dt / self.battery_capacity_J,
        )

        # Emergency violation flag.
        emg_violation = bool(self.delays_s[0] > self.D_th)
        self.violation_history.append(emg_violation)
        if len(self.violation_history) > 50:
            self.violation_history.pop(0)

        # Reward.
        d_emg_ratio = float(self.delays_s[0] / max(self.D_th, EPS))
        r_emg = -float(np.clip(d_emg_ratio, 0.0, 5.0)) ** 2
        r_imp = float(
            math.log1p(min(self.rates_bps[1] / max(self.R_target, EPS), 1.0))
        )
        if self.R_norm_ref is not None and self.R_norm_ref > 0:
            r_norm = float(self.rates_bps[2] / self.R_norm_ref)
        else:
            r_norm = float(self.rates_bps[2] / max(self.rates_bps[2], EPS))
        reward = (
            self.w_emg * r_emg
            + self.w_imp * r_imp
            + self.w_norm * r_norm
            - self.lambda_pen * float(emg_violation)
        )

        self.prev_action = raw
        self._raw_action = raw.copy()
        self.slot += 1
        terminated = self.slot >= self.episode_slots
        truncated = False

        obs = self._get_obs()
        info = {
            "delay_s": self.delays_s,
            "rate_bps": self.rates_bps,
            "queue_bits": self.queue_bits,
            "drops_bits": self.drops_bits,
            "los_state": self.channel.los_state,
            "emergency_violation": emg_violation,
            "ood_flag": self.ood_flag,
            "gain": gain,
            "battery_fraction": self.battery_fraction,
            "slot": self.slot,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ helpers
    def _get_obs(self):
        gain = np.asarray(getattr(self, "_last_gain", None), dtype=float)
        if gain.size == 0:
            gain = np.asarray(self.channel.gain(self.radius_m), dtype=float)
        # Vectorized NumPy for the per-class scalings (fewer Python calls).
        g_obs = np.clip((np.log10(np.maximum(gain, 1e-30)) + 14.0) / 10.0, 0.0, 1.0)
        q_obs = np.clip(np.log1p(self.queue_bits) / self.log1p_cap, 0.0, 1.0)
        d_obs = np.clip(self.delays_s / max(self.D_th, EPS), 0.0, 5.0)
        batt = np.array([self.battery_fraction])
        viol_frac = np.array(
            [float(np.mean(self.violation_history)) if self.violation_history else 0.0]
        )
        obs = np.concatenate(
            [g_obs, q_obs, d_obs, batt, viol_frac, self._raw_action.astype(np.float32)]
        ).astype(np.float32)
        return obs

    # ---- gym API convenience ----
    def seed(self, seed=None):
        self.reset(seed=seed)
