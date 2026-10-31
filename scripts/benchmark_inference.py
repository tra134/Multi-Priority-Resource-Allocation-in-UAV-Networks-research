"""Benchmark actor + ensemble-critic inference time per slot (Claim E, Phase 8).

Measures the fast-loop cost on CPU: one actor forward pass plus the ensemble
disagreement ``uncertainty()`` call, with the same code path used by the
controller.  The plan requires < 5 ms per slot (plan.md decisions D1/E).

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --ckpt results/ckpt/sac_seed0_N10_latest.pt
    python scripts/benchmark_inference.py --n 2000 --device cpu --out results/phase8

Output:
    results/phase8/inference_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.agents.sac import SACAgent


def main():
    parser = argparse.ArgumentParser(
        description="Measure actor + ensemble inference time per slot (Claim E)"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default=None, help="optional trained checkpoint")
    parser.add_argument("--n", type=int, default=1000, help="timed slots")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/phase8")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sac_cfg = cfg.get("sac", {})
    rng = np.random.default_rng(args.seed)
    n_critics = int(sac_cfg.get("N_critics", 10))
    agent = SACAgent(
        obs_dim=17,
        act_dim=6,
        device=args.device,
        seed=args.seed,
        N_critics=n_critics,
        hidden=int((sac_cfg.get("hidden") or [256])[0]),
        rng=rng,
    )
    if args.ckpt and os.path.exists(args.ckpt):
        agent.load(args.ckpt)
        print(f"loaded checkpoint {args.ckpt}")
    elif args.ckpt:
        print(f"WARNING: checkpoint not found: {args.ckpt} (using untrained agent)")

    obs = rng.normal(size=17).astype(np.float32)

    # Warm-up (CUDA kernels, lazy imports, caches).
    for _ in range(args.warmup):
        action = agent.select_action(obs, deterministic=True)
        agent.uncertainty(obs, action)

    per_slot_ms = []
    actor_ms = []
    unc_ms = []
    t_start = time.perf_counter()
    for _ in range(args.n):
        t0 = time.perf_counter()
        action = agent.select_action(obs, deterministic=True)
        t1 = time.perf_counter()
        agent.uncertainty(obs, action)
        t2 = time.perf_counter()
        actor_ms.append((t1 - t0) * 1000.0)
        unc_ms.append((t2 - t1) * 1000.0)
        per_slot_ms.append((t2 - t0) * 1000.0)
    wall = time.perf_counter() - t_start

    def pct(values, q):
        return float(np.percentile(values, q))

    result = {
        "device": args.device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "N_critics": n_critics,
        "n_slots": args.n,
        "warmup": args.warmup,
        "actor_ms": {
            "mean": float(statistics.fmean(actor_ms)),
            "median": float(statistics.median(actor_ms)),
            "p95": pct(actor_ms, 95),
        },
        "uncertainty_ms": {
            "mean": float(statistics.fmean(unc_ms)),
            "median": float(statistics.median(unc_ms)),
            "p95": pct(unc_ms, 95),
        },
        "total_ms_per_slot": {
            "mean": float(statistics.fmean(per_slot_ms)),
            "median": float(statistics.median(per_slot_ms)),
            "p95": pct(per_slot_ms, 95),
            "throughput_slots_per_s": float(args.n / max(wall, 1e-9)),
        },
        "claim_E_target_ms": 5.0,
        "claim_E_passed": bool(statistics.median(per_slot_ms) < 5.0),
        "note": "Inference only; excludes environment step and LLM wall-clock call.",
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "inference_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== inference benchmark (actor + ensemble) ===")
    print(f"device           : {args.device}  N_critics={n_critics}")
    print(f"actor median     : {result['actor_ms']['median']:.3f} ms")
    print(f"uncertainty med. : {result['uncertainty_ms']['median']:.3f} ms")
    print(f"total median     : {result['total_ms_per_slot']['median']:.3f} ms/slot "
          f"(p95 {result['total_ms_per_slot']['p95']:.3f} ms)")
    print(f"Claim E (<5 ms)  : {'PASS' if result['claim_E_passed'] else 'FAIL'}")
    print(f"wrote {path}")
    return 0 if result["claim_E_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
