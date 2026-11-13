"""Evaluate non-learning sanity baselines on LinMECEnv (docs/lin2025_spec.md).

Writes <out>/episodes.csv, <out>/summary.csv and <out>/manifest.json.
--out is required (no default; never point it at results/ files you want to keep).
Default seeds are the validation range (2000..). Test seeds 5000-5999 are refused
unless --allow-test is given (final numbers only).

Example:
  python scripts/evaluate_baselines.py --policies LocalComputing RandomPolicy \\
      --n-sensors 8 --seeds 2000 5 --out /tmp/scratch_eval
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_RANGE = (5000, 5999)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "lin2025.yaml"))
    p.add_argument("--policies", nargs="+", default=["LocalComputing", "RandomPolicy", "NearestUAVFullPower", "GreedyLocalOrOffload"])
    p.add_argument("--n-sensors", nargs="+", type=int, default=[8])
    p.add_argument("--sic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mode", choices=["POS", "FOS"], default="POS")
    p.add_argument("--seeds", nargs=2, type=int, metavar=("START", "COUNT"), default=[2000, 100])
    p.add_argument("--allow-test", action="store_true", help="permit seeds in 5000-5999")
    p.add_argument("--out", required=True, help="output directory (required)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    start, count = args.seeds
    seeds = list(range(start, start + count))
    if not args.allow_test and any(TEST_RANGE[0] <= s <= TEST_RANGE[1] for s in seeds):
        print("refusing: seeds overlap the test range 5000-5999 (use --allow-test for final numbers)", file=sys.stderr)
        return 2

    import pandas as pd
    from uavmec.env.lin_env import LinMECEnv

    from uavmec.baselines.policies import POLICIES
    from uavmec.config import load_config
    from uavmec.eval.metrics import aggregate_episodes, bootstrap_ci
    from uavmec.eval.runner import evaluate

    unknown = [n for n in args.policies if n not in POLICIES]
    if unknown:
        print(f"unknown policies {unknown}; choose from {sorted(POLICIES)}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, summ, cfgs = [], [], {}
    for n in args.n_sensors:
        cfg = load_config(args.config, n_sensors=n)
        cfgs[n] = dataclasses.asdict(cfg)
        for name in args.policies:
            policy = POLICIES[name]()
            df = evaluate(lambda cfg=cfg: LinMECEnv(cfg, sic=args.sic, mode=args.mode), policy, seeds)
            df.insert(0, "n_sensors", n)
            frames.append(df)
            row = aggregate_episodes(df.to_dict("records"))
            row.update(policy=name, n_sensors=n)
            for col in ("success_rate", "mean_latency_ms", "mean_energy_mw"):
                lo, hi = bootstrap_ci(df[col].to_numpy())
                row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = lo, hi
            summ.append(row)
            print(f"N={n} {name}: success={row['success_rate']:.4f} lat={row['mean_latency_ms']:.3f} ms energy={row['mean_energy_mw']:.3f} mW")
    pd.concat(frames).to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    manifest = {
        "git_commit": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "configs": cfgs,
        "sic": args.sic,
        "mode": args.mode,
        "seeds": seeds,
        "policies": args.policies,
        "llm_client": "none",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
