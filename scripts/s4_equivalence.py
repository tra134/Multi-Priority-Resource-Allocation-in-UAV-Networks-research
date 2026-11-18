"""S-4 equivalence analysis (TOST with Welch t) of learned methods at N = 8.

Spec: docs/lin2025_spec.md Section 11, S-4. Descriptive; no new evaluation.
Usage: python scripts/s4_equivalence.py --per-run <csv> --out <new dir>
"""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

MARGIN_PP = 1.0
N_SENSORS = 8
PAIRS = [
    ("POS_MADDPG_NOMA", "FOS_MADDPG_NOMA"),
    ("POS_MADDPG_NOMA", "POS_MAPPO_NOMA"),
    ("POS_MADDPG_NOMA", "POS_MADDPG_noSIC"),
    ("FOS_MADDPG_NOMA", "FOS_MADDPG_noSIC"),
    ("POS_MAPPO_NOMA", "POS_MAPPO_noSIC"),
]


def welch_tost(a, b, margin=MARGIN_PP, alpha=0.05):
    """a, b: 1-D samples in pp. Returns dict for the difference mean(a)-mean(b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    se = np.sqrt(se2)
    diff = a.mean() - b.mean()
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = stats.t.ppf(1 - alpha, df)
    lo, hi = diff - tcrit * se, diff + tcrit * se
    t_lower = (diff + margin) / se  # H0: diff <= -margin
    t_upper = (diff - margin) / se  # H0: diff >= +margin
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    return {
        "mean_a": a.mean(), "std_a": a.std(ddof=1), "n_a": na,
        "mean_b": b.mean(), "std_b": b.std(ddof=1), "n_b": nb,
        "diff": diff, "se": se, "df": df, "ci90_lo": lo, "ci90_hi": hi,
        "t_lower": t_lower, "t_upper": t_upper,
        "p_lower": p_lower, "p_upper": p_upper, "p_tost": max(p_lower, p_upper),
        "equivalent": bool(lo > -margin and hi < margin),
        "smallest_margin": max(abs(lo), abs(hi)),
    }


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"output dir exists: {out}")
    inp = Path(args.per_run)
    rows = list(csv.DictReader(inp.open()))
    rows = [r for r in rows if int(r["n_sensors"]) == N_SENSORS]
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(100.0 * float(r["success_rate"]))
    res = []
    for A, B in PAIRS:
        d = welch_tost(by[A], by[B])
        res.append({"method_a": A, "method_b": B, **d})
    out.mkdir(parents=True)
    with (out / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    manifest = {
        "script": "scripts/s4_equivalence.py",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "input": str(inp),
        "input_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
        "n_sensors": N_SENSORS, "margin_pp": MARGIN_PP, "alpha_per_side": 0.05,
        "timestamp": datetime.datetime.now().isoformat(), "llm_client": "none",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for r in res:
        print(f"{r['method_a']} vs {r['method_b']}: diff={r['diff']:.4f} "
              f"CI90=[{r['ci90_lo']:.4f},{r['ci90_hi']:.4f}] p={r['p_tost']:.4g} "
              f"equiv={r['equivalent']} margin={r['smallest_margin']:.4f}")


if __name__ == "__main__":
    main()
