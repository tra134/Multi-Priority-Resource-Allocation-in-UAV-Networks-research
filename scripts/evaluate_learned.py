"""Evaluate trained learners (runs written by scripts/train.py) greedily on fixed seeds.

Reads <run>/manifest.json and <run>/final.pt, rebuilds env + learner exactly as train.py does,
evaluates with explore=False on seeds SEEDS_START .. SEEDS_START+N_EPISODES-1 (default 5000.. =
test seeds, spec Section 8: final numbers only), and also evaluates Local_Computing on the same
seeds for each N present (paired evaluation, A-36).
Writes to --out: per_run.csv, summary.csv, gate.json (spec Section 7), manifest.json.
--out is required (no default). Success rate is a fraction in [0, 1].

Example:
  python scripts/evaluate_learned.py --runs /tmp/run_a /tmp/run_b --out /tmp/eval_scratch
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METRICS = ("success_rate", "mean_latency_ms", "mean_energy_mw")
LOCAL = "Local_Computing"
N_TARGET = 8

# Paper targets, docs/lin2025_spec.md Section 4 / Table 2 (N = 8 only). Success is a fraction.
# POS_MAPPO_NOMA success uses 71.4% (Fig. 4b/5a and text 85.63/1.20), not Table 2's 61.25% (A-35).
# Energy is None where the paper gives none ("-").
TARGETS = {
    "POS_MADDPG_NOMA": (0.8563, 7.48, 5.68),
    "FOS_MADDPG_NOMA": (0.6088, 8.11, 4.195),
    "POS_MAPPO_NOMA": (0.714, 12.50, 11.41),
    "Local_Computing": (0.6125, 18.76, 18.76),  # A-32
    "POS_MADDPG_noSIC": (0.5982, 13.08, None),
    "FOS_MADDPG_noSIC": (0.4211, 13.50, None),
    "POS_MAPPO_noSIC": (0.3462, 11.38, None),
}
TARGET_NOTES = {"POS_MAPPO_NOMA": "success target 71.4% per A-35 (Table 2 says 61.25%)",
                "Local_Computing": "success target 61.25% per A-32"}
GATE4_TARGET, GATE4_TOL = 0.8563, 0.10  # spec Section 7 item 4


def git_info() -> dict:
    def run(*a):
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories written by scripts/train.py")
    p.add_argument("--out", required=True, help="output directory (required)")
    p.add_argument("--seeds-start", type=int, default=5000, help="first evaluation seed (5000 = test seeds)")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--env-config", default=str(ROOT / "configs" / "lin2025.yaml"))
    return p


def method_name(algo: str, mode: str, sic: bool) -> str:
    return f"{mode}_{algo.upper()}_{'NOMA' if sic else 'noSIC'}"


def eval_learner(env, learner, seeds) -> dict:
    """Same loop and metric functions as train.py's evaluate()."""
    from uavmec.eval.metrics import aggregate_episodes, summarize_outcomes
    rows = []
    for s in seeds:
        obs, info = env.reset(seed=s)
        state, slots = info["state"], []
        while True:
            obs, _r, term, trunc, info = env.step(learner.act(obs, state, explore=False))
            state = info["state"]
            slots.append(info["outcomes"])
            if term or trunc:
                break
        rows.append(summarize_outcomes(slots))
    agg = aggregate_episodes(rows)
    return {k: agg[k] for k in (*METRICS, "n_episodes")}


def mean_std(vals):
    import numpy as np
    a = np.asarray(vals, dtype=float)
    return float(np.mean(a)), (float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)


def gate(summ: dict) -> dict:
    """summ[(method, n)] -> {metric: mean over training seeds}. Spec Section 7, items 1-4."""
    def sr(m, n):
        return summ.get((m, n), {}).get("success_rate")

    def verdict(checks, missing_ok_reason):
        done = [c for c in checks if c["evaluated"]]
        if any(not c["holds"] for c in done):
            status = "fail"
        elif done and len(done) == len(checks):
            status = "pass"
        else:
            status = "not_evaluable"
        reason = ("all comparisons evaluated" if status == "pass" else
                  "at least one evaluated comparison fails" if status == "fail" else missing_ok_reason)
        return {"status": status, "reason": reason, "checks": checks}

    def cmp(a, b, n):  # a > b at n
        x, y = sr(a, n), sr(b, n)
        if x is None or y is None:
            return {"claim": f"{a} > {b} (N={n})", "evaluated": False, "holds": None, "values": [x, y]}
        return {"claim": f"{a} > {b} (N={n})", "evaluated": True, "holds": bool(x > y), "values": [x, y]}

    n = N_TARGET
    # Item 1. FOS-vs-MAPPO ordering is deliberately not used (A-35).
    c1 = [cmp("POS_MADDPG_NOMA", "FOS_MADDPG_NOMA", n), cmp("POS_MADDPG_NOMA", "POS_MAPPO_NOMA", n),
          cmp("FOS_MADDPG_NOMA", "FOS_MADDPG_noSIC", n), cmp("POS_MAPPO_NOMA", "POS_MAPPO_noSIC", n)]
    item1 = verdict(c1, "some compared methods or N = 8 are missing; pass needs every comparison evaluated")
    # Item 2: every NOMA method with a noSIC twin at the same N beats it.
    c2 = []
    for (m, nn) in sorted(summ):
        if m.endswith("_NOMA") and (m[:-4] + "noSIC", nn) in summ:
            c2.append(cmp(m, m[:-4] + "noSIC", nn))
    item2 = verdict(c2, "no NOMA/noSIC pair present at the same N") if c2 else {
        "status": "not_evaluable", "reason": "no NOMA/noSIC pair present at the same N", "checks": []}
    # Item 3: trend over N for learned methods; FOS best at N = 2, 4.
    learned = sorted({m for (m, _n) in summ if m != LOCAL})
    ns_of = {m: sorted(nn for (mm, nn) in summ if mm == m) for m in learned}
    if not learned or any(len(v) < 2 for v in ns_of.values()):
        item3 = {"status": "not_evaluable", "checks": [],
                 "reason": "needs at least two N values for every learned method; got "
                           + json.dumps(ns_of)}
    else:
        c3 = []
        for m in learned:
            s = [summ[(m, nn)]["success_rate"] for nn in ns_of[m]]
            lt = [summ[(m, nn)]["mean_latency_ms"] for nn in ns_of[m]]
            c3.append({"claim": f"{m}: success non-increasing in N", "evaluated": True, "n": ns_of[m],
                       "values": s, "holds": all(b <= a for a, b in itertools.pairwise(s))})
            c3.append({"claim": f"{m}: latency non-decreasing in N", "evaluated": True, "n": ns_of[m],
                       "values": lt, "holds": all(b >= a or math.isnan(a) or math.isnan(b)
                                                  for a, b in itertools.pairwise(lt))})
        for nn in (2, 4):
            fos = sr("FOS_MADDPG_NOMA", nn)
            others = {m: sr(m, nn) for m in learned if m != "FOS_MADDPG_NOMA" and m.endswith("_NOMA")
                      and sr(m, nn) is not None}
            if fos is None or not others:
                c3.append({"claim": f"FOS_MADDPG_NOMA best at N={nn}", "evaluated": False, "holds": None})
            else:
                c3.append({"claim": f"FOS_MADDPG_NOMA best at N={nn}", "evaluated": True,
                           "values": {"FOS_MADDPG_NOMA": fos, **others},
                           "holds": all(fos >= v for v in others.values())})
        item3 = verdict(c3, "FOS-best check not evaluable at N = 2, 4 and the trend checks all hold")
    # Item 4.
    v = sr("POS_MADDPG_NOMA", n)
    if v is None:
        item4 = {"status": "not_evaluable", "reason": "POS_MADDPG_NOMA at N = 8 not present", "values": None}
    else:
        d = abs(v - GATE4_TARGET)
        item4 = {"status": "pass" if d <= GATE4_TOL else "fail", "values": {"success_rate": v, "target": GATE4_TARGET,
                 "abs_diff": d, "tolerance": GATE4_TOL}, "reason": "|success - 85.63%| <= 10 percentage points"}
    return {"1": item1, "2": item2, "3": item3, "4": item4}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    runs = []
    for r in args.runs:
        d = Path(r)
        if not (d / "final.pt").exists():
            print(f"error: {d} has no final.pt (training not finished?)", file=sys.stderr)
            return 2
        if not (d / "manifest.json").exists():
            print(f"error: {d} has no manifest.json", file=sys.stderr)
            return 2
        runs.append(d)

    import numpy as np
    import torch

    from uavmec.baselines.policies import LocalComputing
    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.eval.metrics import aggregate_episodes
    from uavmec.eval.runner import evaluate

    seeds = list(range(args.seeds_start, args.seeds_start + args.n_episodes))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_run, run_info = [], []
    for d in runs:
        man = json.loads((d / "manifest.json").read_text())
        ca = man["cli_args"]
        algo, mode, sic, n, seed = ca["algo"], ca["mode"], bool(ca["sic"]), int(ca["n_sensors"]), int(ca["seed"])
        cfg = load_config(args.env_config, n_sensors=n)
        env = LinMECEnv(cfg, sic=sic, mode=mode)
        if algo == "maddpg":
            from uavmec.agents.maddpg import MADDPG as Learner
        else:
            from uavmec.agents.mappo import MAPPO as Learner
        learner = Learner(env.n_agents, env.obs_dim, env.state_dim, cfg.n_uavs, cfg.n_subcarriers,
                          man["train_config"][algo], mode == "FOS", seed)
        ck = torch.load(d / "final.pt", weights_only=False)
        learner.load_state_dict(ck["learner"])
        res = eval_learner(env, learner, seeds)
        name = method_name(algo, mode, sic)
        per_run.append({"method": name, "n_sensors": n, "seed": seed, "run_dir": str(d), **res,
                        "git_commit": man.get("git_commit", "unknown")})
        run_info.append({"run_dir": str(d), "git_commit": man.get("git_commit", "unknown"),
                         "git_dirty": man.get("git_dirty"), "method": name})
        print(f"{name} N={n} seed={seed}: success={res['success_rate']:.4f}")
    for n in sorted({r["n_sensors"] for r in per_run}):
        cfg = load_config(args.env_config, n_sensors=n)
        df = evaluate(lambda cfg=cfg: LinMECEnv(cfg, sic=True, mode="POS"), LocalComputing(), seeds)
        agg = aggregate_episodes(df.to_dict("records"))
        per_run.append({"method": LOCAL, "n_sensors": n, "seed": "", "run_dir": "",
                        **{k: agg[k] for k in (*METRICS, "n_episodes")}, "git_commit": git_info()["git_commit"]})

    cols = ["method", "n_sensors", "seed", "run_dir", *METRICS, "n_episodes", "git_commit"]
    with (out / "per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_run)

    groups: dict = {}
    for r in per_run:
        groups.setdefault((r["method"], r["n_sensors"]), []).append(r)
    summary_rows, summ = [], {}
    for (m, n), rs in sorted(groups.items()):
        row = {"method": m, "n_sensors": n, "n_seeds": len(rs)}
        for k in METRICS:
            mu, sd = mean_std([r[k] for r in rs])
            row[f"{k}_mean"], row[f"{k}_std"] = mu, sd
        summ[(m, n)] = {k: row[f"{k}_mean"] for k in METRICS}
        t = TARGETS.get(m) if n == N_TARGET else None
        for i, k in enumerate(METRICS):
            row[f"target_{k}"] = "" if t is None or t[i] is None else t[i]
        row["target_note"] = TARGET_NOTES.get(m, "") if t else ""
        summary_rows.append(row)
    scols = ["method", "n_sensors", "n_seeds"] + [f"{k}_{s}" for k in METRICS for s in ("mean", "std")] \
        + [f"target_{k}" for k in METRICS] + ["target_note"]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        w.writerows(summary_rows)

    g = {"items": gate(summ), "seeds": [seeds[0], seeds[-1]],
         "note": "spec Section 7; FOS-vs-MAPPO ordering at N = 8 is not a gate item (A-35)"}
    (out / "gate.json").write_text(json.dumps(g, indent=2, default=str))
    manifest = {**git_info(), "timestamp": datetime.now(timezone.utc).isoformat(), "cli_args": vars(args),
                "seeds": seeds, "run_dirs": run_info, "llm_client": "none",
                "torch": torch.__version__, "numpy": np.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
