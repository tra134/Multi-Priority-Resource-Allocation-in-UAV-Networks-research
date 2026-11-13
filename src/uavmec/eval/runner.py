"""Episode runner and multi-seed evaluation for non-learning policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from uavmec.eval.metrics import summarize_outcomes


def run_episode(env, policy, seed: int) -> dict:
    """Run one episode (reset with `seed`) and return the per-episode metrics dict."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    slots = []
    while True:
        actions = policy.act(env, obs, rng)
        obs, _rewards, terminated, truncated, info = env.step(actions)
        slots.append(info["outcomes"])
        if terminated or truncated:
            break
    row = summarize_outcomes(slots)
    row.update(seed=int(seed), policy=getattr(policy, "name", type(policy).__name__), n_slots=len(slots))
    return row


def evaluate(env_factory: Callable[[], object], policy, seeds: Iterable[int]) -> pd.DataFrame:
    """One row per episode. A fresh env is built once; reset(seed) makes episodes independent."""
    env = env_factory()
    return pd.DataFrame([run_episode(env, policy, s) for s in seeds])
