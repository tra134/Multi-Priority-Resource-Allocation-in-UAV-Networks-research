"""Apply the A-39 pilot rule (docs/lin2025_spec.md Section 6) to finished pilot runs.

Reads <run>/eval_log.csv (validation seeds), <run>/train_log.csv (losses column is JSON) and
<run>/manifest.json of runs written by scripts/train.py. Paper numbers are never used.
Writes pilot_check.json and manifest.json to --out (required, no default). Success is a fraction.

  T1 convergence: Delta = mean(last 3 evaluations) - mean(3 evaluations before them); PASS iff |Delta| <= 2 pp.
  T2 stability:   FAIL if any NaN/inf in logged losses or validation metrics, or best - last success > 5 pp.
  Per run: T1 and T2 pass -> "lock"; T2 fails -> "stop_report"; T1 fails and T2 passes ->
  "extend_to_20000" (only change allowed: episodes 10000 -> 20000), except a run that already has
  20000 episodes -> "stop_report" (at most one extension; confirmed by the HUMAN 2026-11-13).
  Overall: "lock" only if all runs lock; "stop_report" if any run stops; else "extend_to_20000".

Example:
  python scripts/check_pilot.py --run /tmp/pilot_maddpg --run /tmp/pilot_mappo --out /tmp/pilot_check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A-39 constants (pre-registered 2026-11-13)
WINDOW = 3
T1_TOL_PP = 2.0
T2_TOL_PP = 5.0
EXTENDED_EPISODES = 20000
EVAL_KEYS = ("success_rate", "mean_latency_ms", "mean_energy_mw")


def git_info() -> dict:
    def run(*a):
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="train.py output directory (repeatable)")
    p.add_argument("--out", required=True, help="output directory (required)")
    return p


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def check_run(d: Path) -> dict:
    """A-39: T1 and T2 on one run. Raises ValueError if fewer than 2 * WINDOW evaluations."""
    with (d / "eval_log.csv").open() as f:
        ev = list(csv.DictReader(f))
    if len(ev) < 2 * WINDOW:
        raise ValueError(f"{d}: {len(ev)} evaluations in eval_log.csv, need at least {2 * WINDOW}")
    eps = [int(r["episode"]) for r in ev]
    succ = [_num(r["success_rate"]) for r in ev]
    man = json.loads((d / "manifest.json").read_text())
    total = int(man.get("train_config", {}).get("episodes", eps[-1]))

    nan_findings = []
    for r in ev:
        for k in EVAL_KEYS:
            if k in r and not math.isfinite(_num(r[k])):  # A-39 T2: any NaN/inf in validation metrics
                nan_findings.append(f"eval_log episode {r['episode']} {k} = {r[k]}")
    with (d / "train_log.csv").open() as f:
        for r in csv.DictReader(f):
            for k, v in json.loads(r["losses"] or "{}").items():
                if _bad(v):
                    nan_findings.append(f"train_log episode {r['episode']} loss {k} = {v}")

    s_last = sum(succ[-WINDOW:]) / WINDOW
    s_prev = sum(succ[-2 * WINDOW:-WINDOW]) / WINDOW
    delta = s_last - s_prev
    best, last = max(succ), succ[-1]
    t1 = bool(round(abs(delta) * 100, 9) <= T1_TOL_PP)
    best_minus_last_pp = (best - last) * 100
    t2 = bool(not nan_findings and round(best_minus_last_pp, 9) <= T2_TOL_PP)
    notes = []
    if not t2:
        decision = "stop_report"
    elif t1:
        decision = "lock"
    elif total >= EXTENDED_EPISODES:
        decision = "stop_report"
        notes.append("T1 failed at 20000 episodes: at most one extension (A-39); "
                     "stop and report")
    else:
        decision = "extend_to_20000"
    return {"episodes": total, "n_evaluations": len(ev), "Delta": delta, "S_last": s_last, "S_prev": s_prev,
            "eval_episodes_last": eps[-WINDOW:], "eval_episodes_prev": eps[-2 * WINDOW:-WINDOW],
            "best_success": best, "last_success": last, "best_minus_last_pp": best_minus_last_pp,
            "nan_findings": nan_findings, "T1": "pass" if t1 else "fail", "T2": "pass" if t2 else "fail",
            "decision": decision, "notes": notes, "run_git_commit": man.get("git_commit", "unknown")}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    runs = {}
    for r in args.run:
        d = Path(r)
        for n in ("eval_log.csv", "train_log.csv", "manifest.json"):
            if not (d / n).exists():
                print(f"error: {d} has no {n}", file=sys.stderr)
                return 2
        try:
            runs[str(d)] = check_run(d)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    decs = [v["decision"] for v in runs.values()]
    overall = "lock" if all(x == "lock" for x in decs) else (
        "stop_report" if "stop_report" in decs else "extend_to_20000")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_check.json").write_text(json.dumps(
        {"rule": "A-39", "decision": overall, "runs": runs,
         "note": "validation log and train losses only; paper numbers not used"}, indent=2, default=str))
    manifest = {**git_info(), "timestamp": datetime.now(timezone.utc).isoformat(), "cli_args": vars(args),
                "input_runs": {k: v["run_git_commit"] for k, v in runs.items()}, "llm_client": "none"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"decision": overall, **{k: v["decision"] for k, v in runs.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
