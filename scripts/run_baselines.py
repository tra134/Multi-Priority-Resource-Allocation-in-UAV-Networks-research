"""Phase 2: baseline comparison (no RL).

Runs strict_priority, fair_share, greedy_sum_rate on the nominal and surge
scenarios, checks the Phase 2 acceptance claims, and saves:

  results/phase2/baselines.csv   - one row per (scenario, policy) with metrics
  results/phase2/baseline_delay.png - Emergency delay over time per baseline

Also prints the nominal load calibration (total arrival / mean achievable
capacity), which should be about 50-60% (plan.md Section 4).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml

from uavrl.env.uav_env import UAVEnv
from uavrl.env.scenarios import Scenario
from uavrl.baselines.static_policies import POLICIES, shares_to_action
from uavrl.eval.runner import run_episode, run_grid

OUT = "results/phase2"
CONFIG = "configs/default.yaml"


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def mean_achievable_capacity(env, n=400):
    """Mean sum-rate under fair share over n slots (nominal channel)."""
    env.reset(seed=0)
    total = 0.0
    for _ in range(n):
        action = env.action_space.sample()  # any action; we override below
        # force fair shares
        action = shares_to_action([1 / 3] * 3, [1 / 3] * 3)
        obs, _, terminated, truncated, info = env.step(action)
        total += float(np.sum(info["rate_bps"]))
        if terminated or truncated:
            env.reset(seed=0)
    return total / n


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()

    # ---- calibration: nominal load vs mean achievable capacity ----
    env_cap = UAVEnv(config=cfg)
    capacity = mean_achievable_capacity(env_cap)
    total_arrival = sum(
        [
            cfg["traffic"]["emergency_mean_mbps"],
            cfg["traffic"]["important_mean_mbps"],
            cfg["traffic"]["normal_mean_mbps"],
        ]
    ) * 1e6
    load = total_arrival / capacity
    print(f"mean achievable capacity (fair share, nominal): {capacity/1e6:.2f} Mbps")
    print(f"total nominal arrival: {total_arrival/1e6:.2f} Mbps")
    print(f"nominal load: {load*100:.1f}%  (target 50-60%)")

    # ---- scenarios ----
    scen_cfg = {
        "nominal": Scenario("nominal", {"type": "none"}, ood=False),
        "surge": Scenario(
            "surge",
            {
                "type": "surge",
                "surge_multiplier": 4.0,
                "duration_slots": 400,
                "start_slot": 500,
            },
            ood=True,
        ),
    }

    rows = []
    for sname, scenario in scen_cfg.items():
        env = UAVEnv(config=cfg, scenario=scenario)
        for pname, policy in POLICIES.items():
            summaries = run_grid(env, policy, n_episodes=20, seed0=1000)
            row = {
                "scenario": sname,
                "policy": pname,
                "episodes": len(summaries),
            }
            for k in [
                "sos_violation_rate",
                "sos_p99_delay_s",
                "mean_reward",
                "mean_rate_emg_bps",
                "mean_rate_imp_bps",
                "mean_rate_norm_bps",
                "total_drops_emg_bits",
                "total_drops_imp_bits",
                "total_drops_norm_bits",
            ]:
                vals = [s[k] for s in summaries]
                row[f"mean_{k}"] = float(np.mean(vals))
                row[f"std_{k}"] = float(np.std(vals))
            rows.append(row)
            print(
                f"{sname:10s} {pname:16s} "
                f"SOS viol={row['mean_sos_violation_rate']*100:5.1f}%  "
                f"p99 delay={row['mean_sos_p99_delay_s']*1000:6.1f} ms"
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
    print("saved", os.path.join(OUT, "baselines.csv"))

    # ---- claims ----
    nom = df[df.scenario == "nominal"].set_index("policy")
    sur = df[df.scenario == "surge"].set_index("policy")
    assert (
        nom.loc["strict_priority", "mean_sos_violation_rate"]
        < nom.loc["greedy_sum_rate", "mean_sos_violation_rate"]
    ), "greedy_sum_rate must violate Emergency more than strict_priority (nominal)"
    assert (
        sur.loc["fair_share", "mean_sos_violation_rate"]
        > sur.loc["strict_priority", "mean_sos_violation_rate"]
    ), "fair_share must violate more than strict_priority (surge)"
    print("Phase 2 acceptance claims verified.")

    # ---- figure: Emergency delay over time (one episode per baseline, nominal) ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (pname, policy) in zip(axes, POLICIES.items()):
        env = UAVEnv(config=cfg, scenario=scen_cfg["nominal"])
        steps, _ = run_episode(env, policy, seed=7)
        slots = [s["slot"] for s in steps]
        delay = [s["delay_emg_s"] * 1000.0 for s in steps]
        ax.plot(slots, delay, lw=1.5)
        ax.axhline(15, color="red", ls="--", lw=1, label="D_th = 15 ms")
        ax.set_title(pname)
        ax.set_xlabel("Slot")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Emergency delay [ms]")
            ax.legend(fontsize=8)
    fig.suptitle("Emergency delay over one nominal episode (baselines)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(OUT, "baseline_delay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
