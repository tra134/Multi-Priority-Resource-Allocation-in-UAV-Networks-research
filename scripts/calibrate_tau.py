"""Phase 5: calibrate the ensemble-uncertainty trigger threshold tau_unc.

Loads a trained SAC agent, runs N in-distribution (nominal) validation
episodes, collects U_t = ensemble std of Q, and sets tau_unc to the q-th
percentile (default 99).  Then evaluates OOD detection (AUROC, precision,
recall, detection lead time) on the surge/shadow/compound/unseen scenarios
using the ground-truth ood_flag (scoring only).

Usage:
    python scripts/calibrate_tau.py --ckpt results/ckpt/sac_seed0_N10_step300000.pt \
        --config configs/default.yaml --quantile 99 --n-val 50

Without --ckpt an untrained agent is used (illustrative only).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.env.uav_env import UAVEnv
from uavrl.env.scenarios import Scenario
from uavrl.agents.sac import SACAgent
from uavrl.uncertainty.calibration import (
    collect_trajectory,
    normalize_U,
    percentile_U,
    ood_metrics,
    auroc,
)

EVAL_SCENARIOS = {
    "surge": {
        "disturbance": {
            "type": "surge",
            "surge_multiplier": 4.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "shadow": {
        "disturbance": {
            "type": "shadow",
            "extra_db": 15.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "compound": {
        "disturbance": {
            "type": "compound",
            "surge_multiplier": 4.0,
            "shadow_db": 15.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "unseen": {
        "disturbance": {
            "type": "compound",
            "surge_multiplier": 8.0,
            "shadow_db": 25.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "nominal": {
        "disturbance": {"type": "none"},
        "ood": False,
    },
}


def make_scenario(name, rng):
    spec = EVAL_SCENARIOS[name]
    return Scenario(name, spec["disturbance"], ood=spec["ood"], rng=rng)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate tau_unc (ensemble uncertainty trigger threshold)"
    )
    parser.add_argument("--ckpt", default=None, help="trained SAC checkpoint (.pt)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quantile", type=float, default=99.0)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/phase5")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(args.seed)
    obs_dim = 17
    act_dim = 6
    agent = SACAgent(
        obs_dim, act_dim, device="cpu", seed=args.seed,
        N_critics=int(cfg.get("sac", {}).get("N_critics", 10)),
        rng=rng,
    )
    if args.ckpt and os.path.exists(args.ckpt):
        agent.load(args.ckpt)
        print(f"loaded checkpoint {args.ckpt}")
    else:
        print("WARNING: no checkpoint -- using untrained agent (illustrative)")

    # ---- in-distribution calibration (nominal, no domain randomization) ----
    val_env = UAVEnv(config=cfg, seed=args.seed + 1)
    val_trajs = []
    for i in range(args.n_val):
        traj = collect_trajectory(val_env, agent, deterministic=True)
        val_trajs.append(traj)

    for traj in val_trajs:
        for r in traj:
            r["U_norm"] = r["U"] / (abs(r["meanQ"]) + 1.0)

    tau_raw = percentile_U(val_trajs, q=args.quantile, normalized=False)
    tau_norm = percentile_U(val_trajs, q=args.quantile, normalized=True)
    print(f"tau_unc (raw U, {args.quantile}th pct) = {tau_raw:.4f}")
    print(f"tau_unc (norm U, {args.quantile}th pct) = {tau_norm:.4f}")

    # ---- OOD evaluation ----
    results = {}
    for sname, spec in EVAL_SCENARIOS.items():
        env = UAVEnv(
            config=cfg,
            scenario=make_scenario(sname, np.random.default_rng(args.seed + 100)),
            seed=args.seed + 200,
        )
        traj = collect_trajectory(env, agent, deterministic=True)
        for r in traj:
            r["U_norm"] = r["U"] / (abs(r["meanQ"]) + 1.0)
        m_raw = ood_metrics(traj, tau_raw, normalized=False)
        m_norm = ood_metrics(traj, tau_norm, normalized=True)
        results[sname] = {
            "auroc_raw": auroc(m_raw["U"], m_raw["ood"].astype(int)),
            "auroc_norm": auroc(m_norm["U"], m_norm["ood"].astype(int)),
            "precision_raw": m_raw["precision"],
            "recall_raw": m_raw["recall"],
            "lead_time_raw": m_raw["lead_time"],
            "precision_norm": m_norm["precision"],
            "recall_norm": m_norm["recall"],
            "lead_time_norm": m_norm["lead_time"],
            "ood_slots": int(m_raw["ood"].sum()),
            "n_slots": len(traj),
        }
        print(
            f"{sname:9s} AUROC_raw={results[sname]['auroc_raw']:.3f} "
            f"AUROC_norm={results[sname]['auroc_norm']:.3f} "
            f"lead_raw={m_raw['lead_time']} lead_norm={m_norm['lead_time']}"
        )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "calibration.txt"), "w") as f:
        f.write(f"tau_unc_raw={tau_raw}\ntau_unc_norm={tau_norm}\n")
        f.write(f"quantile={args.quantile} n_val={args.n_val}\n")
        for sname, r in results.items():
            f.write(f"{sname}: {r}\n")

    # write calibrated trigger config
    trigger_cfg = {
        "trigger": {
            "tau_unc": tau_raw,
            "tau_unc_normalized": tau_norm,
            "tau_unc_quantile": args.quantile,
            "note": "calibrated on in-distribution nominal validation episodes",
        }
    }
    import yaml as _yaml
    with open(os.path.join(args.out, "trigger_calibrated.yaml"), "w") as f:
        _yaml.safe_dump(trigger_cfg, f)
    print("saved", os.path.join(args.out, "trigger_calibrated.yaml"))


if __name__ == "__main__":
    main()
