"""Multi-UAV MEC environment of Lin et al. 2025 (docs/lin2025_spec.md, Section 9)."""

from __future__ import annotations

import numpy as np

from uavmec.config import LinConfig
from uavmec.env import energy
from uavmec.env.channel import channel_gain
from uavmec.env.local import local_latency
from uavmec.env.noma import noma_links
from uavmec.env.s3_variants import S3Variant, as_variant, s3_offload_links
from uavmec.env.tasks import sample_tasks
from uavmec.env.types import SensorAction, TaskOutcome

# Observation normalization constants (so features are roughly in [0, 1]).
_TASK_NORM_BITS = 1.5e6  # largest task size C_n (P: Table 1)
_DEADLINE_NORM_S = 0.050  # largest deadline D_n (P: Table 1)
_SENSOR_BATT_NORM_J = 400.0e3  # largest sensor battery (P: Table 1)
_UAV_BATT_NORM_J = 500.0e3  # largest UAV battery (P: Table 1)


class LinMECEnv:
    """Sensors (agents) choose local computing or offloading to a UAV over NOMA.

    mode "POS": target 0 (local) allowed. mode "FOS": target must be 1..K, else ValueError.
    sic=False gives the noSIC variant. One episode is cfg.n_slots slots; each slot every
    sensor has one task. Public read-only attributes (valid after reset, updated by step):
    cfg, t, sensor_pos (N,2), uav_pos (K,2), task_sizes (N,) bits, task_deadlines (N,) s,
    battery (N,) J, uav_battery (K,) J, channel_gain (N,K).
    """

    def __init__(
        self,
        cfg: LinConfig,
        sic: bool = True,
        mode: str = "POS",
        s3_variant: S3Variant | dict | None = None,
    ):
        if mode not in ("POS", "FOS"):
            raise ValueError(f"mode must be POS or FOS, got {mode!r}")
        self.cfg = cfg
        self.sic = sic
        self.mode = mode
        self.s3 = as_variant(s3_variant)  # S-3 sensitivity variant; None = v1.0 (default)
        self.n_agents = cfg.n_sensors
        # own S_n = (x, y, E, C, D) plus S_k = (x, y, H, B_k, E_k) of every UAV
        self.obs_dim = 5 + 5 * cfg.n_uavs
        self.state_dim = 5 * cfg.n_sensors + 5 * cfg.n_uavs
        self._rng = np.random.default_rng(0)
        self.t = 0
        self.sensor_pos = np.zeros((cfg.n_sensors, 2))
        self.uav_pos = np.zeros((cfg.n_uavs, 2))
        self.task_sizes = np.zeros(cfg.n_sensors)
        self.task_deadlines = np.zeros(cfg.n_sensors)
        self.battery = np.zeros(cfg.n_sensors)
        self.uav_battery = np.zeros(cfg.n_uavs)
        self.channel_gain = np.zeros((cfg.n_sensors, cfg.n_uavs))

    # ------------------------------------------------------------------ helpers
    def _sample_uav_positions(self) -> np.ndarray:
        """Positions in the area with pairwise distance >= d_s (rejection sampling)."""
        cfg = self.cfg
        pos = np.zeros((cfg.n_uavs, 2))
        for k in range(cfg.n_uavs):
            for _ in range(1000):
                cand = self._rng.uniform(0.0, cfg.area_m, size=2)
                if all(np.linalg.norm(cand - pos[j]) >= cfg.d_safe_m for j in range(k)):
                    break
            pos[k] = cand
        return pos

    def _new_slot(self) -> None:
        self.task_sizes, self.task_deadlines = sample_tasks(self._rng, self.cfg.n_sensors, self.cfg)
        self.channel_gain = channel_gain(
            self.sensor_pos, self.uav_pos, self.cfg.uav_height_m, self.cfg.beta0
        )
        if self.s3 is not None:  # S-3 A1-const (other variants: unchanged)
            self.channel_gain = self.s3.channel_gain(self.channel_gain)

    def _state_vector(self) -> np.ndarray:
        cfg = self.cfg
        sensors = np.stack(
            [
                self.sensor_pos[:, 0] / cfg.area_m,
                self.sensor_pos[:, 1] / cfg.area_m,
                self.battery / _SENSOR_BATT_NORM_J,
                self.task_sizes / _TASK_NORM_BITS,
                self.task_deadlines / _DEADLINE_NORM_S,
            ],
            axis=1,
        )
        uavs = np.stack(
            [
                self.uav_pos[:, 0] / cfg.area_m,
                self.uav_pos[:, 1] / cfg.area_m,
                np.full(cfg.n_uavs, cfg.uav_height_m / cfg.uav_height_m),
                np.full(cfg.n_uavs, cfg.subcarrier_bw_hz / cfg.bandwidth_hz),  # B_k
                self.uav_battery / _UAV_BATT_NORM_J,
            ],
            axis=1,
        )
        self._sensors_feat, self._uavs_feat = sensors, uavs
        return np.concatenate([sensors.ravel(), uavs.ravel()]).astype(np.float32)

    def _observations(self) -> tuple[list[np.ndarray], np.ndarray]:
        state = self._state_vector()
        uav_part = self._uavs_feat.ravel()
        obs = [
            np.concatenate([self._sensors_feat[n], uav_part]).astype(np.float32)
            for n in range(self.cfg.n_sensors)
        ]
        return obs, state

    # ---------------------------------------------------------------------- API
    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], dict]:
        """Start a new episode. Same seed gives the same trajectory."""
        cfg = self.cfg
        self._rng = np.random.default_rng(seed)
        self.t = 0
        self.sensor_pos = self._rng.uniform(0.0, cfg.area_m, size=(cfg.n_sensors, 2))
        self.uav_pos = self._sample_uav_positions()
        lo, hi = cfg.sensor_battery_j
        self.battery = self._rng.uniform(lo, hi, size=cfg.n_sensors)  # A-10
        lo, hi = cfg.uav_battery_j
        self.uav_battery = self._rng.uniform(lo, hi, size=cfg.n_uavs)  # A-10
        self._new_slot()
        obs, state = self._observations()
        return obs, {"state": state}

    def _check_actions(self, actions: list[SensorAction]) -> None:
        cfg = self.cfg
        if len(actions) != cfg.n_sensors:
            raise ValueError(f"expected {cfg.n_sensors} actions, got {len(actions)}")
        for a in actions:
            if not 0 <= a.target <= cfg.n_uavs:
                raise ValueError(f"target must be in 0..{cfg.n_uavs}, got {a.target}")
            if self.mode == "FOS" and a.target == 0:
                raise ValueError("FOS mode does not allow target 0 (local computing)")
            if a.target > 0 and not 0 <= a.subcarrier < cfg.n_subcarriers:
                raise ValueError(f"subcarrier must be in 0..{cfg.n_subcarriers - 1}")
            if not (0.0 <= a.power_frac <= 1.0 and 0.0 <= a.bw_frac <= 1.0):
                raise ValueError("power_frac and bw_frac must be in [0, 1]")

    def step(
        self, actions: list[SensorAction]
    ) -> tuple[list[np.ndarray], np.ndarray, bool, bool, dict]:
        """Run one slot. Returns obs, rewards (N,), terminated=False, truncated, info."""
        cfg = self.cfg
        self._check_actions(actions)
        n_s = cfg.n_sensors
        outcomes: list[TaskOutcome | None] = [None] * n_s

        # --- offloading users: NOMA rates (Eq. 5-7) ---
        off = [n for n in range(n_s) if actions[n].target > 0]
        extra_s = None  # S-3 UAV-cb compute latency of each offloader (None = v1.0)
        if off:
            uav_idx = np.array([actions[n].target - 1 for n in off])
            sub_idx = np.array([actions[n].subcarrier for n in off])
            power = np.array([actions[n].power_frac * cfg.p_max_w for n in off])
            if self.s3 is None:
                bw = np.array([actions[n].bw_frac * cfg.subcarrier_bw_hz for n in off])
                rx = self.channel_gain[off, uav_idx] * power
                sinr, rate, decoded = noma_links(
                    uav_idx, sub_idx, rx, bw, cfg.noise_w, cfg.rate_req_bps, self.sic
                )
            else:  # S-3 variant path (V0 reproduces the branch above exactly)
                sinr, rate, decoded, extra_s = s3_offload_links(
                    self.s3, cfg, self.sic, np.array(off), uav_idx, sub_idx,
                    self.channel_gain[off, uav_idx], power,
                    np.array([actions[n].bw_frac for n in off]), self.task_sizes[off],
                )

        spent = np.zeros(n_s)
        for n in range(n_s):
            bits, ddl, batt = self.task_sizes[n], self.task_deadlines[n], self.battery[n]
            if actions[n].target == 0:  # local, Eq. 1-2
                lat = float(local_latency(bits, cfg.c_cyc, cfg.f_sensor_hz))
                req = float(energy.local_energy_required(bits, cfg.c_cyc, cfg.kappa, cfg.f_sensor_hz))
                spent[n] = float(energy.local_energy(req, batt))
                reason = "deadline" if lat > ddl else ("energy" if req > batt else None)
                outcomes[n] = TaskOutcome(
                    bits, ddl, "local", None, None, None, None, None, lat, spent[n],
                    reason is None, reason,
                )
            else:  # offload, Eq. 7-8
                i = off.index(n)
                r = float(rate[i])
                lat = bits / r if r > 0 else float("inf")
                req = float(energy.offload_energy_required(power[i], lat, cfg.slot_s))
                if extra_s is not None:  # S-3 UAV-cb: compute time counts for the deadline, not energy
                    lat = lat + float(extra_s[i])
                # A-20: spent energy = min(required, battery); a task failing for energy drains the battery.
                spent[n] = min(req, batt)
                if not decoded[i]:
                    # A-19: under SIC decode failure == rate failure (A-7); priority decode/rate > deadline > energy.
                    reason = "decode" if self.sic else "rate"
                elif lat > ddl:
                    reason = "deadline"
                elif req > batt:
                    reason = "energy"
                else:
                    reason = None
                outcomes[n] = TaskOutcome(
                    bits, ddl, "offload", int(uav_idx[i]), int(sub_idx[i]), float(sinr[i]), r,
                    bool(decoded[i]), lat, spent[n], reason is None, reason,
                )

        rewards = np.array(
            [cfg.reward_success if o.success else cfg.reward_fail for o in outcomes], dtype=float
        )
        self.battery = energy.battery_update(self.battery, spent)  # Eq. 10
        uav_j = energy.uav_hover_energy(cfg.n_uavs, cfg.uav_hover_power_w, cfg.slot_s)  # A-13, A-37
        self.uav_battery = np.maximum(self.uav_battery - uav_j, 0.0)
        self.t += 1
        self._new_slot()
        obs, state = self._observations()
        info = {"outcomes": outcomes, "state": state, "uav_energy_j": uav_j}
        return obs, rewards, False, self.t >= cfg.n_slots, info
