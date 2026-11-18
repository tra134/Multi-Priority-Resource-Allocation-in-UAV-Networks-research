"""Recompute gate.json from an existing per_run.csv (written by scripts/evaluate_learned.py).

No model is loaded and no episode is evaluated: it only re-applies the gate rules (spec Section 7)
to the stored per-run numbers. --out must not exist (nothing is overwritten).
Writes gate.json and manifest.json to --out.

Example:
  python scripts/recompute_gate.py --per-run results/eval_X/per_run.csv --out results/eval_X_gate_fix
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("evaluate_learned", HERE / "evaluate_learned.py")
el = importlib.util.module_from_spec(spec)
spec.loader.exec_module(el)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-run", required=True, help="per_run.csv from evaluate_learned.py")
    p.add_argument("--out", required=True, help="new output directory (must not exist)")
    a = p.parse_args(argv)
    src, out = Path(a.per_run), Path(a.out)
    if out.exists():
        print(f"error: {out} already exists; refusing to overwrite", file=sys.stderr)
        return 2
    rows = [r for r in csv.DictReader(src.open()) if r["method"] != el.LOCAL]
    groups: dict = {}
    per_seed: dict = {}
    for r in rows:
        n = int(r["n_sensors"])
        vals = {k: float(r[k]) for k in el.METRICS}
        groups.setdefault((r["method"], n), []).append(vals)
        per_seed.setdefault((r["method"], n), {})[int(r["seed"])] = {
            "success_rate": vals["success_rate"], "mean_latency_ms": vals["mean_latency_ms"]}
    summ = {k: {m: el.mean_std([v[m] for v in vs])[0] for m in el.METRICS} for k, vs in groups.items()}
    g = {"items": el.gate(summ, per_seed), "recomputed_from": str(src),
         "note": "spec Section 7; recomputed from stored per_run.csv, no test seed evaluated"}
    out.mkdir(parents=True)
    (out / "gate.json").write_text(json.dumps(g, indent=2, default=str))
    manifest = {**el.git_info(), "timestamp": datetime.now(timezone.utc).isoformat(), "input_csv": str(src),
                "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "llm_client": "none",
                "cli_args": vars(a)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
