"""Calibrate c_cyc (C-1) and kappa (C-2) on LocalComputing (docs/lin2025_spec.md, Section 5).

C-1: grid search c_cyc so the mean latency of successful LocalComputing tasks is
     closest to 18.76 ms.
C-2: kappa in closed form (energy is linear in kappa) so LocalComputing mean sensor
     energy per slot equals 18.76 mW, using the chosen c_cyc.
Seeds are fixed to 1000-1099 (calibration range). The LocalComputing success rate is
reported only as a validation number against 61.25% (A-32); it is never tuned.
This script does NOT edit configs/lin2025.yaml; it prints the values.
--out is required.
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

TARGET_LATENCY_MS = 18.76
TARGET_ENERGY_MW = 18.76
SEEDS = list(range(1000, 1100))


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "lin2025.yaml"))
    p.add_argument("--c-grid", nargs=3, type=float, metavar=("START", "STOP", "STEP"), default=[1.0, 200.0, 1.0],
                   help="coarse c_cyc grid (cycles/bit); a fine grid of step/10 around the best follows")
    p.add_argument("--out", required=True, help="output directory (required)")
    args = p.parse_args(argv)

    import numpy as np
    import pandas as pd

    from uavmec.baselines.policies import LocalComputing
    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.eval.metrics import aggregate_episodes
    from uavmec.eval.runner import evaluate

    base = load_config(args.config)

    def measure(**ov):
        cfg = load_config(args.config, **ov)
        df = evaluate(lambda: LinMECEnv(cfg, sic=True, mode="POS"), LocalComputing(), SEEDS)
        return aggregate_episodes(df.to_dict("records"))

    def sweep(values):
        rows = []
        for c in values:
            r = measure(c_cyc=float(c))
            rows.append({"c_cyc": float(c), "mean_latency_ms": r["mean_latency_ms"], "success_rate": r["success_rate"],
                         "mean_energy_mw": r["mean_energy_mw"], "abs_err_ms": abs(r["mean_latency_ms"] - TARGET_LATENCY_MS)})
        return rows

    start, stop, step = args.c_grid
    rows = sweep(np.arange(start, stop + 1e-9, step))
    best = min((r for r in rows if not np.isnan(r["abs_err_ms"])), key=lambda r: r["abs_err_ms"])
    fine = np.arange(best["c_cyc"] - step, best["c_cyc"] + step + 1e-9, step / 10)
    fine = [c for c in fine if c > 0]
    rows += sweep(fine)
    best = min((r for r in rows if not np.isnan(r["abs_err_ms"])), key=lambda r: r["abs_err_ms"])
    c_star = best["c_cyc"]

    # C-2: energy is linear in kappa (battery clip ignored; checked by the verification run below)
    r0 = measure(c_cyc=c_star)
    kappa0 = float(base.kappa)
    kappa_star = kappa0 * TARGET_ENERGY_MW / r0["mean_energy_mw"]
    rv = measure(c_cyc=c_star, kappa=kappa_star)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "grid.csv", index=False)
    chosen = {
        "c_cyc": c_star,
        "kappa": kappa_star,
        "latency_ms_at_c_cyc": best["mean_latency_ms"],
        "energy_mw_with_kappa": rv["mean_energy_mw"],
        "latency_ms_with_kappa": rv["mean_latency_ms"],
        "local_success_rate": rv["success_rate"],
        "a32_local_success_target": 0.6125,
        "note": "success rate is a validation number, not tuned",
    }
    (out / "chosen.json").write_text(json.dumps(chosen, indent=2))
    manifest = {
        "git_commit": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "config": dataclasses.asdict(base),
        "seeds": "1000-1099",
        "targets": {"latency_ms": TARGET_LATENCY_MS, "energy_mw": TARGET_ENERGY_MW},
        "llm_client": "none",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(chosen, indent=2))
    print("configs/lin2025.yaml was NOT edited; apply c_cyc and kappa after HUMAN approval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
