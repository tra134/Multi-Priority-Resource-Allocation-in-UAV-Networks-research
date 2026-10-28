"""Experiment runner: evaluate a policy over N episodes under a scenario.

Returns per-step logs (list of dicts) and an aggregated summary per episode.
Used by Phase 2 baseline comparison and by later phases.
"""

from __future__ import annotations

import numpy as np


def run_episode(env, policy, seed=None):
    """Run a single episode; return (steps: list[dict], summary: dict)."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    info = {}
    steps = []
    done = False
    while not done:
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(
            {
                "slot": info["slot"],
                "reward": reward,
                "emergency_violation": bool(info["emergency_violation"]),
                "ood_flag": bool(info["ood_flag"]),
                "delay_emg_s": float(info["delay_s"][0]),
                "delay_imp_s": float(info["delay_s"][1]),
                "delay_norm_s": float(info["delay_s"][2]),
                "rate_emg_bps": float(info["rate_bps"][0]),
                "rate_imp_bps": float(info["rate_bps"][1]),
                "rate_norm_bps": float(info["rate_bps"][2]),
                "queue_emg_bits": float(info["queue_bits"][0]),
                "queue_imp_bits": float(info["queue_bits"][1]),
                "queue_norm_bits": float(info["queue_bits"][2]),
                "drops_emg_bits": float(info["drops_bits"][0]),
                "drops_imp_bits": float(info["drops_bits"][1]),
                "drops_norm_bits": float(info["drops_bits"][2]),
                "gain_emg": float(info["gain"][0]),
                "gain_imp": float(info["gain"][1]),
                "gain_norm": float(info["gain"][2]),
                "battery_fraction": float(info["battery_fraction"]),
            }
        )
        done = terminated or truncated
    summary = _summarize(steps)
    return steps, summary


def _summarize(steps):
    if not steps:
        return {}
    n = len(steps)
    arr = {k: np.array([s[k] for s in steps]) for k in steps[0] if k != "slot"}
    sos = arr["emergency_violation"]
    return {
        "episodes": 1,
        "slots": n,
        "mean_reward": float(np.mean(arr["reward"])),
        "sos_violation_rate": float(np.mean(sos)),
        "sos_p99_delay_s": float(np.percentile(arr["delay_emg_s"], 99)),
        "mean_delay_emg_s": float(np.mean(arr["delay_emg_s"])),
        "mean_rate_emg_bps": float(np.mean(arr["rate_emg_bps"])),
        "mean_rate_imp_bps": float(np.mean(arr["rate_imp_bps"])),
        "mean_rate_norm_bps": float(np.mean(arr["rate_norm_bps"])),
        "total_drops_emg_bits": float(np.sum(arr["drops_emg_bits"])),
        "total_drops_imp_bits": float(np.sum(arr["drops_imp_bits"])),
        "total_drops_norm_bits": float(np.sum(arr["drops_norm_bits"])),
        "final_battery_fraction": float(arr["battery_fraction"][-1]),
    }


def run_grid(env, policy, n_episodes=20, seed0=0):
    """Run n_episodes; return list of per-episode summaries."""
    summaries = []
    for i in range(n_episodes):
        _, summary = run_episode(env, policy, seed=seed0 + i)
        summaries.append(summary)
    return summaries
