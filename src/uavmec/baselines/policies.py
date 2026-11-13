"""Non-learning SANITY baselines (not the paper's learned methods; see docs/lin2025_spec.md).

`LocalComputing` is the paper's Local_Computing baseline. The others only bound what
a trivial controller achieves. None of them reads ood flags or scenario labels.
"""

from __future__ import annotations

import numpy as np

from uavmec.env.types import SensorAction


def _n_uavs(env) -> int:
    return len(env.uav_pos)


def _n_sub(env) -> int:
    cfg = env.cfg
    for name in ("n_subcarriers", "M", "subcarriers", "m_subcarriers"):
        if hasattr(cfg, name):
            return int(getattr(cfg, name))
    raise AttributeError("cannot find the number of subcarriers in env.cfg")


def _cfg(env, *names):
    for n in names:
        if hasattr(env.cfg, n):
            return float(getattr(env.cfg, n))
    raise AttributeError(f"env.cfg has none of {names}")


class Policy:
    name = "Policy"

    def act(self, env, obs, rng: np.random.Generator) -> list[SensorAction]:
        raise NotImplementedError


class LocalComputing(Policy):
    """The paper's Local_Computing baseline: every task is computed on the sensor."""

    name = "LocalComputing"

    def act(self, env, obs, rng):
        return [SensorAction(0, 0, 0.0, 0.0) for _ in range(len(obs))]


class RandomPolicy(Policy):
    """Uniform random target, subcarrier, power and bandwidth fraction (sanity baseline).

    In FOS mode targets are drawn from 1..K only. `fos=None` reads env.mode at act time.
    """

    name = "RandomPolicy"

    def __init__(self, fos: bool | None = None):
        self.fos = fos

    def act(self, env, obs, rng):
        fos = self.fos if self.fos is not None else getattr(env, "mode", "POS") == "FOS"
        k, m = _n_uavs(env), _n_sub(env)
        lo = 1 if fos else 0
        return [
            SensorAction(
                int(rng.integers(lo, k + 1)),
                int(rng.integers(0, m)),
                float(rng.uniform()),
                float(rng.uniform()),
            )
            for _ in range(len(obs))
        ]


def _nearest_uav(env) -> np.ndarray:
    d = np.linalg.norm(np.asarray(env.sensor_pos)[:, None, :] - np.asarray(env.uav_pos)[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def _offload_all(env, nearest, mask) -> dict[int, SensorAction]:
    """Offload sensors in `mask` to their nearest UAV, round-robin subcarriers per UAV."""
    m = _n_sub(env)
    counters: dict[int, int] = {}
    out = {}
    for n in range(len(nearest)):
        if not mask[n]:
            continue
        k = int(nearest[n])
        c = counters.get(k, 0)
        counters[k] = c + 1
        out[n] = SensorAction(k + 1, c % m, 1.0, 1.0)
    return out


class NearestUAVFullPower(Policy):
    """Offload everything to the nearest UAV, full power and bandwidth (sanity baseline)."""

    name = "NearestUAVFullPower"

    def act(self, env, obs, rng):
        nearest = _nearest_uav(env)
        acts = _offload_all(env, nearest, [True] * len(nearest))
        return [acts[n] for n in range(len(nearest))]


class GreedyLocalOrOffload(Policy):
    """Compute locally if T^l <= D_n, otherwise offload to the nearest UAV at full power.

    Sanity baseline. Uses env.cfg c_cyc and F_n (A-4: T^l = C_n * c_cyc / F_n).
    """

    name = "GreedyLocalOrOffload"

    def act(self, env, obs, rng):
        c_cyc = _cfg(env, "c_cyc", "cycles_per_bit")
        f_n = _cfg(env, "f_sensor_hz", "sensor_cpu_hz", "F_n", "f_n", "sensor_cpu")
        t_local = np.asarray(env.task_sizes) * c_cyc / f_n
        local = t_local <= np.asarray(env.task_deadlines)
        if getattr(env, "mode", "POS") == "FOS":
            local = np.zeros_like(local)  # FOS forbids local computing
        nearest = _nearest_uav(env)
        off = _offload_all(env, nearest, ~local)
        return [SensorAction(0, 0, 0.0, 0.0) if local[n] else off[n] for n in range(len(nearest))]


POLICIES = {
    c.name: c for c in (LocalComputing, RandomPolicy, NearestUAVFullPower, GreedyLocalOrOffload)
}
