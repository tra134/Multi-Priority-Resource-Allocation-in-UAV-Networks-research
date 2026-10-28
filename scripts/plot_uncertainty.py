"""Phase 5: plot U_t timelines with OOD windows shaded.

Usage:
    python scripts/plot_uncertainty.py --ckpt results/ckpt/... --config configs/default.yaml

Saves results/phase5/uncertainty_timeline_<scenario>.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.env.uav_env import UAVEnv
from uavrl.env.scenarios import Scenario
from uavrl.agents.sac import SACAgent
from uavrl.uncertainty.calibration import collect_trajectory, normalize_U, ood_metrics

SCENARIOS = {
    "nominal": {"disturbance": {"type": "none"}, "ood": False},
    "surge": {
        "disturbance": {"type": "surge", "surge_multiplier": 4.0,
                        "duration_slots": 400, "start_slot": [500, 1200]},
        "ood": True,
    },
    "shadow": {
        "disturbance": {"type": "shadow", "extra_db": 15.0,
                        "duration_slots": 400, "start_slot": [500, 1200]},
        "ood": True,
    },
    "compound": {
        "disturbance": {"type": "compound", "surge_multiplier": 4.0,
                        "shadow_db": 15.0, "duration_slots": 400,
                        "start_slot": [500, 1200]},
        "ood": True,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Plot U_t timelines with OOD windows")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/phase5")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(args.seed)
    agent = SACAgent(
        17, 6, device="cpu", seed=args.seed,
        N_critics=int(cfg.get("sac", {}).get("N_critics", 10)), rng=rng,
    )
    if args.ckpt and os.path.exists(args.ckpt):
        agent.load(args.ckpt)

    tau = None
    cal_path = os.path.join(args.out, "trigger_calibrated.yaml")
    if os.path.exists(cal_path):
        import yaml as _yaml
        tc = _yaml.safe_load(open(cal_path))
        tau = tc["trigger"]["tau_unc"]

    os.makedirs(args.out, exist_ok=True)
    for sname, spec in SCENARIOS.items():
        env = UAVEnv(
            config=cfg,
            scenario=Scenario(sname, spec["disturbance"], ood=spec["ood"],
                              rng=np.random.default_rng(args.seed + 1)),
            seed=args.seed + 2,
        )
        traj = collect_trajectory(env, agent, deterministic=True)
        slots = [r["slot"] for r in traj]
        U = np.array([r["U"] for r in traj])
        ood = np.array([r["ood_flag"] for r in traj])
        delay = np.array([r["delay_emg_s"] for r in traj]) * 1000.0

        fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
        ax = axes[0]
        ax.plot(slots, U, label="U_t (ensemble std Q)", color="#1f77b4")
        if tau is not None:
            ax.axhline(tau, color="red", ls="--", label=f"tau_unc={tau:.3f}")
        ax.set_ylabel("U_t")
        ax.set_title(f"{sname} — ensemble uncertainty U_t")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(slots, delay, label="Emergency delay [ms]", color="#d62728")
        ax.axhline(15, color="red", ls="--", label="D_th = 15 ms")
        # shade OOD window
        ood_idx = np.where(ood)[0]
        if len(ood_idx):
            ax.axvspan(slots[ood_idx[0]], slots[ood_idx[-1]],
                       color="orange", alpha=0.15, label="OOD window")
        ax.set_xlabel("Slot")
        ax.set_ylabel("Delay [ms]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()
        path = os.path.join(args.out, f"uncertainty_timeline_{sname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
