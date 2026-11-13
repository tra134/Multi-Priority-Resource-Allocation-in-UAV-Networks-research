"""Phase 2 analysis: turn run_grid.py metrics CSVs into the protocol Section 4 tables.

Implements docs/rq4_phase2_protocol.md Section 4 and nothing else:
  confirmatory_holm.csv        12 pre-declared comparisons (M12 vs M3, M4, M7, M13 on S1, S2, S4),
                               metric sos_violation_ood, Holm family fixed at 12
  throughput_loss_vs_m3.csv    Important+Normal response-phase throughput, M12 vs M3
  exploratory.csv              M12c / M8 / M9 / m7_rule_shield_fitted rows, NOT in the Holm family
  nan_counts.csv               NaN rows per method, scenario and column
  llm_cost.csv                 per LLM method and scenario, mean and CI over training seeds
  missing_comparisons.csv      only written when something is missing
  manifest.json                inputs with sha256, git commit, timestamp, n_boot, seed

Refuses to run if any LLM-method row is not client == "real" (MockLLM is never evidence,
CLAUDE.md Rule 4), unless --allow-mock, which stamps every output with the MOCK note.
Exits non-zero if any of the 12 comparisons has no rows, unless --allow-missing.
--out is required; nothing is written under results/ unless you point it there.

Example:
    venv/bin/python scripts/analyze_phase2.py --in /tmp/p2/metrics.csv --out /tmp/p2_analysis
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavrl.eval.stats import (  # noqa: E402
    MAIN_BASELINES,
    MAIN_METHOD,
    MAIN_SCENARIOS,
    MOCK_NOTE,
    comparison_table,
    hierarchical_bootstrap_ci,
    holm_bonferroni,
    paired_comparison,
)

VIOLATION = "sos_violation_ood"
THROUGHPUT_COLS = ("delivered_imp_ood_mbps", "delivered_norm_ood_mbps")
THROUGHPUT = "impnorm_ood_mbps"  # derived column
M3 = "m3_pure_sac"
LLM_METHODS = ("m12_proposed", "m12c_instant_rule", "m8_random_llm", "m9_periodic_llm")
NON_LLM_METHODS = ("m3_pure_sac", "m4_redq_sac", "m7_rule_shield", "m13_fixed_reserve",
                   "m7_rule_shield_fitted")
FORBIDDEN_METHODS = ("m11_always_llm",)
FAMILY_SIZE = len(MAIN_BASELINES) * len(MAIN_SCENARIOS)  # 12
TARGET_VIOLATION_REL = 0.40   # at least 40 % lower than M3
TARGET_THROUGHPUT_LOSS = 0.10  # at most 10 % loss
TARGET_CALL_RATE = 0.08       # upper end of "5-8 % of slots"
EXPLORATORY_NOTE = "exploratory - not Holm-corrected, no confirmatory claim"

# (main, baselines) pairs of Section 4 "Secondary, exploratory"
EXPLORATORY_PAIRS = (
    ("m12c_instant_rule", ("m3_pure_sac", "m4_redq_sac", "m7_rule_shield",
                           "m13_fixed_reserve", "m12_proposed")),
    ("m12_proposed", ("m8_random_llm", "m9_periodic_llm", "m7_rule_shield_fitted")),
    ("m12c_instant_rule", ("m8_random_llm", "m9_periodic_llm")),
)

COST_COLUMNS = (
    "llm_calls", "llm_call_rate", "prompt_tokens", "completion_tokens", "llm_valid",
    "llm_invalid", "llm_timeouts", "llm_latency_ms_median", "measured_latency_ms_median",
    "inference_ms_per_slot", "wall_ms_per_slot", "mask_active_fraction",
    "fallback_active_fraction", "diagnosis_accuracy", "trigger_precision",
)
NAN_COLUMNS = (VIOLATION,) + THROUGHPUT_COLS + COST_COLUMNS


class AnalysisError(Exception):
    """Raised when an input check fails; the run must not produce tables."""


# --------------------------------------------------------------------------- checks
def check_clients(df, allow_mock=False):
    """Refuse mock LLM rows (Rule 4). Returns True if mock rows were allowed through."""
    if "client" not in df.columns:
        raise AnalysisError("input has no 'client' column; cannot verify real LLM rows")
    bad_methods = sorted(set(df["method"]) & set(FORBIDDEN_METHODS))
    if bad_methods:
        raise AnalysisError(f"forbidden methods present in Phase 2 input: {bad_methods}")
    llm = df[df["method"].isin(LLM_METHODS)]
    non_real = llm[llm["client"] != "real"]
    non_llm = df[df["method"].isin(NON_LLM_METHODS)]
    non_none = non_llm[non_llm["client"] != "none"]
    mock = bool(len(non_real) or len(non_none))
    if mock and not allow_mock:
        detail = sorted(set(zip(non_real["method"], non_real["client"], strict=True)))
        raise AnalysisError(
            f"{len(non_real)} LLM-method row(s) are not client == 'real' {detail}; "
            f"{len(non_none)} non-LLM row(s) are not client == 'none'. MockLLM is never "
            "evidence (CLAUDE.md Rule 4). Use --allow-mock only for pipeline testing.")
    return mock


def check_duplicates(df):
    keys = ["method", "scenario", "seed", "episode", "eval_seed"]
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise AnalysisError(f"{int(dup.sum())} rows share (method, scenario, seed, episode, "
                            "eval_seed); inputs overlap or a run was appended twice")


# --------------------------------------------------------------------------- tables
def add_throughput(df):
    missing = [c for c in THROUGHPUT_COLS if c not in df.columns]
    if missing:
        raise AnalysisError(f"throughput columns missing from input: {missing}")
    df = df.copy()
    df[THROUGHPUT] = df[THROUGHPUT_COLS[0]] + df[THROUGHPUT_COLS[1]]
    return df


def _cell(df, method, scenario, metric):
    sub = df[(df["method"] == method) & (df["scenario"] == scenario)]
    return sub.dropna(subset=[metric])


def holm_fixed_family(p_values, family_size, alpha=0.05):
    """Holm over a family of fixed size: missing comparisons count as p = 1."""
    p = list(p_values)
    pad = max(0, family_size - len(p))
    adj, _ = holm_bonferroni(np.array(p + [1.0] * pad), alpha)
    adj = adj[:len(p)]
    return adj, adj <= alpha


def confirmatory_table(df, n_boot, seed):
    """The 12 pre-declared comparisons. Returns (table, missing list)."""
    base = df.dropna(subset=[VIOLATION])
    tab = comparison_table(base, metric=VIOLATION, main=MAIN_METHOD,
                           baselines=MAIN_BASELINES, scenarios=MAIN_SCENARIOS,
                           n_boot=n_boot, seed=seed)
    present = (set(zip(tab["scenario"], tab["baseline"], strict=True))
               if len(tab) else set())
    missing = [{"scenario": sc, "main": MAIN_METHOD, "baseline": b}
               for sc in MAIN_SCENARIOS for b in MAIN_BASELINES if (sc, b) not in present]
    if len(tab):
        adj, rej = holm_fixed_family(tab["p_value"], FAMILY_SIZE)
        tab["p_holm"] = adj
        tab["reject_holm"] = rej
        tab["family_size"] = FAMILY_SIZE
        tab["family"] = "confirmatory"
        if (tab["note"] != "").any():
            raise AnalysisError("a confirmatory row carries a mock note; refusing")
    return tab, missing


def throughput_table(df, n_boot, seed):
    """Loss = (T_M3 - T_M12) / T_M3 on paired episodes; positive loss means M12 lost
    throughput. Also reports the two slide targets against M3."""
    df = add_throughput(df)
    rows = []
    for sc in MAIN_SCENARIOS:
        a = _cell(df, MAIN_METHOD, sc, THROUGHPUT)
        b = _cell(df, M3, sc, THROUGHPUT)
        if a.empty or b.empty:
            continue
        r = paired_comparison(a, b, THROUGHPUT, n_boot=n_boot, seed=seed)
        row = {"scenario": sc, "main": MAIN_METHOD, "baseline": M3, "metric": THROUGHPUT,
               "mean_m12": r["mean_a"], "mean_m3": r["mean_b"],
               "loss_pct": 100 * r["rel_improvement"],
               "loss_ci_low_pct": 100 * r["rel_ci_low"],
               "loss_ci_high_pct": 100 * r["rel_ci_high"],
               "throughput_target_max_loss_pct": 100 * TARGET_THROUGHPUT_LOSS,
               "throughput_target_met": bool(r["rel_improvement"] <= TARGET_THROUGHPUT_LOSS),
               "n_pairs": r["n_pairs"], "n_seeds": r["n_seeds"], "n_boot": r["n_boot"]}
        v = _cell(df, MAIN_METHOD, sc, VIOLATION)
        v3 = _cell(df, M3, sc, VIOLATION)
        if len(v) and len(v3):
            rv = paired_comparison(v, v3, VIOLATION, n_boot=n_boot, seed=seed)
            row["violation_reduction_pct"] = 100 * rv["rel_improvement"]
            row["violation_target_min_reduction_pct"] = 100 * TARGET_VIOLATION_REL
            row["violation_target_met"] = bool(rv["rel_improvement"] >= TARGET_VIOLATION_REL)
        rows.append(row)
    return pd.DataFrame(rows)


def exploratory_table(df, n_boot, seed):
    """Exploratory comparisons; Holm columns are dropped on purpose."""
    base = df.dropna(subset=[VIOLATION])
    frames, missing = [], []
    for main, baselines in EXPLORATORY_PAIRS:
        tab = comparison_table(base, metric=VIOLATION, main=main, baselines=baselines,
                               scenarios=MAIN_SCENARIOS, n_boot=n_boot, seed=seed)
        present = (set(zip(tab["scenario"], tab["baseline"], strict=True))
                   if len(tab) else set())
        missing += [{"scenario": sc, "main": main, "baseline": b}
                    for sc in MAIN_SCENARIOS for b in baselines if (sc, b) not in present]
        if len(tab):
            tab = tab.drop(columns=["p_holm", "reject_holm"])
            frames.append(tab)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["family"] = "exploratory"
        out["note"] = (out["note"].where(out["note"] == "", out["note"] + "; ")
                       + EXPLORATORY_NOTE)
    return out, missing


def nan_counts(df):
    cols = [c for c in NAN_COLUMNS if c in df.columns]
    rows = []
    for (m, sc), g in df.groupby(["method", "scenario"], sort=True):
        for c in cols:
            rows.append({"method": m, "scenario": sc, "column": c, "n_rows": len(g),
                         "n_nan": int(g[c].isna().sum())})
    return pd.DataFrame(rows)


def cost_table(df, n_boot, seed):
    """Mean and 95 % CI over training seeds per LLM method, scenario, column."""
    cols = [c for c in COST_COLUMNS if c in df.columns]
    rows = []
    for m in LLM_METHODS:
        for sc in MAIN_SCENARIOS:
            sub = df[(df["method"] == m) & (df["scenario"] == sc)]
            if sub.empty:
                continue
            for c in cols:
                s = sub.dropna(subset=[c])
                if s.empty:
                    rows.append({"method": m, "scenario": sc, "column": c, "mean": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "n_seeds": 0,
                                 "n_rows": 0})
                    continue
                r = hierarchical_bootstrap_ci(s, c, n_boot=n_boot, seed=seed)
                rows.append({"method": m, "scenario": sc, "column": c, "mean": r["mean"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                             "n_seeds": r["n_seeds"], "n_rows": len(s)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- driver
def analyze(df, n_boot=10000, seed=12345, allow_mock=False):
    """Run every table. Returns (tables dict, missing confirmatory, missing exploratory,
    mock flag, notes). Raises AnalysisError if the mock check fails."""
    mock = check_clients(df, allow_mock)
    check_duplicates(df)
    df = add_throughput(df)
    confirm, miss_c = confirmatory_table(df, n_boot, seed) if not mock else \
        _confirmatory_mock(df, n_boot, seed)
    explo, miss_e = exploratory_table(df, n_boot, seed)
    tables = {
        "confirmatory_holm": confirm,
        "throughput_loss_vs_m3": throughput_table(df, n_boot, seed),
        "exploratory": explo,
        "nan_counts": nan_counts(df),
        "llm_cost": cost_table(df, n_boot, seed),
    }
    if mock:
        for t in tables.values():
            if len(t):
                t["mock_note"] = MOCK_NOTE
    return tables, miss_c, miss_e, mock


def _confirmatory_mock(df, n_boot, seed):
    """Same as confirmatory_table but tolerates the mock note (pipeline testing only)."""
    base = df.dropna(subset=[VIOLATION])
    tab = comparison_table(base, metric=VIOLATION, n_boot=n_boot, seed=seed)
    present = (set(zip(tab["scenario"], tab["baseline"], strict=True))
               if len(tab) else set())
    missing = [{"scenario": sc, "main": MAIN_METHOD, "baseline": b}
               for sc in MAIN_SCENARIOS for b in MAIN_BASELINES if (sc, b) not in present]
    if len(tab):
        adj, rej = holm_fixed_family(tab["p_value"], FAMILY_SIZE)
        tab["p_holm"], tab["reject_holm"] = adj, rej
        tab["family_size"] = FAMILY_SIZE
        tab["family"] = "confirmatory"
    return tab, missing


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load(inputs):
    frames = []
    for p in inputs:
        p = Path(p)
        f = p / "metrics.csv" if p.is_dir() else p
        if not f.exists():
            raise SystemExit(f"input not found: {f}")
        frames.append((f, pd.read_csv(f)))
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="run_grid.py metrics.csv files (or directories holding one)")
    ap.add_argument("--out", required=True, help="output directory (no default)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--allow-mock", action="store_true",
                    help="pipeline testing only; stamps every output with the MOCK note")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write tables even if some of the 12 comparisons have no rows")
    args = ap.parse_args(argv)

    loaded = load(args.inputs)
    df = pd.concat([d for _, d in loaded], ignore_index=True)
    try:
        tables, miss_c, miss_e, mock = analyze(df, args.n_boot, args.seed, args.allow_mock)
    except AnalysisError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if miss_c:
        print(f"MISSING: {len(miss_c)} of {FAMILY_SIZE} confirmatory comparisons have no rows:",
              file=sys.stderr)
        for m in miss_c:
            print(f"  {m['scenario']} {m['main']} vs {m['baseline']}", file=sys.stderr)
        if not args.allow_missing:
            print("Not writing tables (use --allow-missing to override).", file=sys.stderr)
            return 2
    if miss_e:
        print(f"note: {len(miss_e)} exploratory comparison(s) have no rows", file=sys.stderr)
    if miss_c or miss_e:
        pd.DataFrame([{**m, "family": "confirmatory"} for m in miss_c]
                     + [{**m, "family": "exploratory"} for m in miss_e]
                     ).to_csv(out / "missing_comparisons.csv", index=False)
    for name, tab in tables.items():
        tab.to_csv(out / f"{name}.csv", index=False)
    manifest = {
        "inputs": [{"path": str(f), "sha256": sha256_of(f), "rows": len(d)}
                   for f, d in loaded],
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_boot": args.n_boot, "seed": args.seed,
        "holm_family_size": FAMILY_SIZE,
        "mock": mock, "mock_note": MOCK_NOTE if mock else None,
        "missing_confirmatory": miss_c, "missing_exploratory": miss_e,
        "allow_missing": args.allow_missing,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(tables)} tables to {out}" + (f"  [{MOCK_NOTE}]" if mock else ""))
    return 2 if miss_c and not args.allow_missing else 0


if __name__ == "__main__":
    sys.exit(main())
