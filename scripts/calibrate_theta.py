"""Phase 7/9: tune the Emergency-queue trigger threshold ``theta_queue``.

The hybrid controller fires when ``Q_emg > theta_queue`` for ``debounce_m``
slots.  This script sweeps candidate thresholds with the **queue-only** trigger
(no uncertainty term), so the result is independent of ``tau_unc``.

Validation protocol
-------------------
* Tune on validation seeds (default ``--seed0 20000``), never on the Phase 9
  test seeds (10000 + seed * 1000 + episode).
* The mock LLM uses the *nominal* latency; S3/S4 latency degradation is not
  mixed into the threshold choice.
* Pick the lowest-call-rate threshold whose mean SOS violation rate is at or
  below ``--target-sos``.  If no threshold reaches the target, report the one
  with the lowest violation rate and say so.

Usage
-----
    python scripts/calibrate_theta.py \
        --ckpt /content/drive/MyDrive/uavrl/ckpt/sac_seed0_N10/sac_seed0_N10_latest.pt \
        --n-critics 10 --scenarios S1,S2,S4 --episodes 3 \
        --theta 2000,5000,10000,20000,35000,50000 \
        --out results/phase9/theta_sweep.csv

Output: per-episode CSV + a printed summary and recommendation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_grid import SCENARIOS, make_scenario  # noqa: E402

from uavrl.agents.sac import SACAgent  # noqa: E402
from uavrl.control.hybrid_controller import HybridController  # noqa: E402
from uavrl.env.uav_env import UAVEnv  # noqa: E402
from uavrl.llm.cache import LLMCache  # noqa: E402
from uavrl.llm.client import MockLLMClient  # noqa: E402

FIELDS = [
    "theta_queue", "scenario", "seed", "episode",
    "sos_violation_rate", "mean_delay_emg_s", "mean_reward",
    "llm_calls", "llm_call_rate", "trigger_fraction",
    "mask_active_fraction", "fallback_active_fraction",
]


def parse_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep theta_queue for the queue-only trigger"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-critics", type=int, default=10)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenarios", default="S1,S2,S4")
    parser.add_argument("--theta", default="2000,5000,10000,20000,35000,50000")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=20000,
                        help="validation seed base (keep away from grid seeds)")
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--target-sos", type=float, default=0.01)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--ttl-slots", type=int, default=None)
    parser.add_argument("--out", default="results/phase9/theta_sweep.csv")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    timing = cfg.get("timing", {})
    slots = int(args.slots or timing.get("episode_slots", 2000))
    cfg.setdefault("timing", {})["episode_slots"] = slots
    latency = float(timing.get("llm_latency_median_ms", 350.0))
    latency_sigma = float(timing.get("llm_latency_sigma", 0.6))
    timeout_ms = int(timing.get("llm_timeout_ms", 5000))
    trigger_cfg = cfg.get("trigger", {})
    cooldown = int(args.cooldown if args.cooldown is not None
                   else trigger_cfg.get("cooldown", 50))
    ttl_slots = int(args.ttl_slots if args.ttl_slots is not None
                    else trigger_cfg.get("ttl_slots", 600))
    fallback_slots = int(cfg.get("fallback", {}).get("fallback_slots", 200))

    thetas = [float(t) for t in parse_list(args.theta)]
    scenarios = parse_list(args.scenarios)
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}")

    agent = SACAgent(
        obs_dim=17, act_dim=6, device="cpu", seed=0,
        N_critics=args.n_critics, rng=np.random.default_rng(0),
    )
    agent.load(args.ckpt)
    print(f"loaded {args.ckpt}")
    print(f"sweep theta={thetas} on {scenarios} "
          f"({args.episodes} episodes, validation seeds from {args.seed0})")

    rows = []
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for theta in thetas:
        for scenario_id in scenarios:
            for episode in range(args.episodes):
                seed = args.seed0 + episode
                env = UAVEnv(
                    config=cfg,
                    scenario=make_scenario(scenario_id, seed),
                    seed=seed,
                )
                client = MockLLMClient(
                    latency_median_ms=latency,
                    latency_sigma=latency_sigma,
                    rng=np.random.default_rng(seed),
                    cache=LLMCache(enabled=False),
                )
                ctrl = HybridController(
                    agent, env, client, cfg,
                    trigger_mode="queue",
                    tau_unc=None,
                    theta_queue=theta,
                    debounce_m=int(trigger_cfg.get("debounce_m", 3)),
                    cooldown=cooldown,
                    ttl_slots=ttl_slots,
                    timeout_ms=timeout_ms,
                    fallback_slots=fallback_slots,
                    rng=np.random.default_rng(seed),
                )
                steps, summary = ctrl.run_episode(seed=seed)
                row = {
                    "theta_queue": theta,
                    "scenario": scenario_id,
                    "seed": seed,
                    "episode": episode,
                    "sos_violation_rate": summary["sos_violation_rate"],
                    "mean_delay_emg_s": summary["mean_delay_emg_s"],
                    "mean_reward": summary["mean_reward"],
                    "llm_calls": len(ctrl.call_log),
                    "llm_call_rate": len(ctrl.call_log) / max(len(steps), 1),
                    "trigger_fraction": float(np.mean(
                        [s["trigger"] for s in steps]
                    )),
                    "mask_active_fraction": summary["mask_active_fraction"],
                    "fallback_active_fraction":
                        summary["fallback_active_fraction"],
                }
                rows.append(row)
                print(
                    f"theta={theta:7.0f} {scenario_id} ep={episode} "
                    f"SOS={row['sos_violation_rate']*100:5.2f}% "
                    f"calls={row['llm_call_rate']*100:4.2f}% "
                    f"trigger={row['trigger_fraction']*100:5.1f}% "
                    f"mask={row['mask_active_fraction']*100:5.1f}%",
                    flush=True,
                )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary + recommendation
    print("\n=== summary (mean over validation episodes) ===")
    summary_rows = []
    for theta in thetas:
        subset = [r for r in rows if r["theta_queue"] == theta]
        mean_sos = float(np.mean([r["sos_violation_rate"] for r in subset]))
        mean_calls = float(np.mean([r["llm_call_rate"] for r in subset]))
        mean_mask = float(np.mean([r["mask_active_fraction"] for r in subset]))
        summary_rows.append((theta, mean_sos, mean_calls, mean_mask))
        print(f"theta={theta:7.0f}  SOS={mean_sos*100:5.2f}%  "
              f"calls={mean_calls*100:4.2f}%  mask={mean_mask*100:5.1f}%")

    eligible = [r for r in summary_rows if r[1] <= args.target_sos]
    if eligible:
        best = min(eligible, key=lambda r: (r[2], -r[0]))
        print(f"\nrecommended theta_queue = {best[0]:.0f} "
              f"(SOS {best[1]*100:.2f}% <= target "
              f"{args.target_sos*100:.1f}%, lowest call rate)")
    else:
        best = min(summary_rows, key=lambda r: r[1])
        print(f"\nno threshold reached target SOS {args.target_sos*100:.1f}%; "
              f"lowest violation is theta_queue = {best[0]:.0f} "
              f"(SOS {best[1]*100:.2f}%)")

    print(f"\nper-episode rows -> {out_path}")


if __name__ == "__main__":
    main()
