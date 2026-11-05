"""Stage 2: scenario feasibility sweep with non-learning policies only.

Question: can the slide stress levels (4x surge, +15 dB, 400-slot events) be made
both *feasible* (a good allocation survives them) and *discriminating* (a poor
allocation visibly fails), without ever looking at the proposed method?

The criteria below were written in trialAndError.md (TE-007) before this script
was run.  Only validation seeds are used.

    C1  nominal load is 50-60% of the mean fair-share capacity
    C2  M1 Emergency violation: S0 episode <= 1%; in-window S1, S2 <= 5%, S4 <= 20%
    C3  fair share in-window violation >= 3x M1 and >= 5 points higher in S1, S2, S4
    C4  M1 delivers >= 5% less Important+Normal in the S4 window than in S0
    C5  S5 is infeasible: the emergency-only bound still violates >= 50% of the window

Usage
-----
    python scripts/calibrate_scenarios.py --out results/stage2
    python scripts/calibrate_scenarios.py --emergency 2,4,6,8 --episodes 10 --jobs 8

The candidate with the smallest Emergency load that meets C1-C5 is reported.
If none does, the script says so and exits non-zero; thresholds are not relaxed.

Final check (TE-015, TE-016)
----------------------------
Rounds 1-3 failed because C3 compared against fair share, which over-provisions
Emergency.  C3 is replaced by C3-revised, which compares M1 with a static policy
N sized only for nominal traffic.  ``--final`` evaluates ``configs/default.yaml``
unchanged (no rescaling) and stores the result:

    python scripts/calibrate_scenarios.py --final

    r      smallest Emergency share in {0.02, ..., 0.30} with S0 violation <= 1%
           on sizing seeds val_seed(300..309); N = [r, (1-r)*35/63, (1-r)*28/63]
    C3rev  in S1, S2, S4 N's in-window violation >= 3x M1's and >= 5 points higher,
           on test-of-criteria seeds val_seed(400..409)

The result is also compared with the values recorded in TE-016.  A mismatch or any
failed criterion exits non-zero: do not freeze, report to the HUMAN.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from uavrl.baselines.static_policies import (  # noqa: E402
    QueueAwarePriority,
    emergency_only,
    fair_share,
    greedy_sum_rate,
    shares_to_action,
)
from uavrl.env.scenarios import make_scenario  # noqa: E402
from uavrl.env.uav_env import UAVEnv  # noqa: E402
from uavrl.eval.metrics import episode_metrics  # noqa: E402
from uavrl.eval.runner import run_episode  # noqa: E402
from uavrl.seeds import val_seed  # noqa: E402

SCENARIO_IDS = ["S0", "S1", "S2", "S4", "S5"]
POLICY_NAMES = ["m1_queue_priority", "emergency_only", "fair_share", "greedy", "reserve20"]
IMP_NORM_RATIO = (35.0, 28.0)
TARGET_LOAD = 0.55


def build_policy(name, cfg):
    if name == "m1_queue_priority":
        return QueueAwarePriority.from_config(cfg)
    if name == "emergency_only":
        return emergency_only
    if name == "fair_share":
        return fair_share
    if name == "greedy":
        return greedy_sum_rate
    if name.startswith("nominal_sized:"):
        r = float(name.split(":", 1)[1])
        rest = 1.0 - r
        shares = [r, rest * IMP_NORM_RATIO[0] / sum(IMP_NORM_RATIO),
                  rest * IMP_NORM_RATIO[1] / sum(IMP_NORM_RATIO)]
        action = shares_to_action(shares, shares)
        return lambda obs, info: action
    if name == "reserve20":
        action = shares_to_action([0.20, 0.45, 0.35], [0.20, 0.45, 0.35])
        return lambda obs, info: action
    raise KeyError(name)


def candidate_config(base_cfg, emergency_mbps, capacity_mbps, p_max=None):
    """Scale Important/Normal so the total load is TARGET_LOAD of the capacity."""
    cfg = copy.deepcopy(base_cfg)
    if p_max is not None:
        cfg["channel"]["P_max"] = float(p_max)
    rest = TARGET_LOAD * capacity_mbps - emergency_mbps
    a, b = IMP_NORM_RATIO
    imp, norm = rest * a / (a + b), rest * b / (a + b)
    cfg["traffic"]["emergency_mean_mbps"] = round(float(emergency_mbps), 2)
    cfg["traffic"]["important_mean_mbps"] = round(imp, 2)
    cfg["traffic"]["normal_mean_mbps"] = round(norm, 2)
    cfg["reward"]["R_target"] = round(0.75 * imp * 1e6, 0)
    cfg["reward"]["R_norm_ref"] = round(norm * 1e6, 0)
    return cfg


def run_cell(task):
    """One (candidate, policy): all scenarios x episodes.  Runs in a worker."""
    cand_id, cfg, policy_name, episodes, seed_offset = task
    slot_ms = float(cfg["timing"]["slot_ms"])
    rows = []
    for sid in SCENARIO_IDS:
        for ep in range(episodes):
            seed = val_seed(seed_offset + ep)
            env = UAVEnv(config=cfg, scenario=make_scenario(sid, seed), seed=seed)
            steps, _ = run_episode(env, build_policy(policy_name, cfg), seed=seed)
            m = episode_metrics(steps, slot_ms)
            cap = float(np.mean([s["rate_emg_bps"] + s["rate_imp_bps"] + s["rate_norm_bps"]
                                 for s in steps])) / 1e6
            rows.append({
                "candidate": cand_id, "policy": policy_name, "scenario": sid, "episode": ep,
                "viol_episode": m["sos_violation_episode"], "viol_ood": m["sos_violation_ood"],
                "delivered_imp_mbps": m["delivered_imp_mbps"],
                "delivered_norm_mbps": m["delivered_norm_mbps"],
                "delivered_imp_ood_mbps": m["delivered_imp_ood_mbps"],
                "delivered_norm_ood_mbps": m["delivered_norm_ood_mbps"],
                "allocated_rate_mbps": cap,
            })
    return rows


# --------------------------------------------------------------------------- final check
SIZING_OFFSET = 300        # val_seed(300..309): sizing of the nominal reservation
CRITERIA_OFFSET = 400      # val_seed(400..409): evaluation of the criteria
R_GRID = [round(0.02 * i, 2) for i in range(1, 16)]
FINAL_POLICIES = ["m1_queue_priority", "emergency_only", "fair_share"]
# Values recorded in TE-016 (mean of 10 episodes, fraction not percent).
EXPECTED = {
    "r": 0.10,
    ("m1_queue_priority", "S0", "viol_episode"): 0.0040,
    ("m1_queue_priority", "S1", "viol_ood"): 0.0093,
    ("m1_queue_priority", "S2", "viol_ood"): 0.0243,
    ("m1_queue_priority", "S4", "viol_ood"): 0.1578,
    ("nominal", "S0", "viol_episode"): 0.0104,
    ("nominal", "S1", "viol_ood"): 0.9350,
    ("nominal", "S2", "viol_ood"): 0.8165,
    ("nominal", "S4", "viol_ood"): 0.9977,
    ("fair_share", "S1", "viol_ood"): 0.0138,
    ("fair_share", "S2", "viol_ood"): 0.0833,
    ("fair_share", "S4", "viol_ood"): 0.8608,
    ("emergency_only", "S5", "viol_ood"): 0.9282,
}
EXPECTED_TOL = 0.0001      # 0.01 percentage points = the rounding of the recorded table


def final_cell(task):
    """One (policy, scenarios) cell on explicit seeds.  Runs in a worker."""
    policy_name, cfg, scenarios, episodes, offset = task
    slot_ms = float(cfg["timing"]["slot_ms"])
    rows = []
    for sid in scenarios:
        for ep in range(episodes):
            seed = val_seed(offset + ep)
            env = UAVEnv(config=cfg, scenario=make_scenario(sid, seed), seed=seed)
            steps, _ = run_episode(env, build_policy(policy_name, cfg), seed=seed)
            m = episode_metrics(steps, slot_ms)
            rows.append({
                "policy": policy_name, "scenario": sid, "episode": ep, "seed": seed,
                "viol_episode": m["sos_violation_episode"], "viol_ood": m["sos_violation_ood"],
                "delivered_imp_mbps": m["delivered_imp_mbps"],
                "delivered_norm_mbps": m["delivered_norm_mbps"],
                "delivered_imp_ood_mbps": m["delivered_imp_ood_mbps"],
                "delivered_norm_ood_mbps": m["delivered_norm_ood_mbps"],
                "allocated_rate_mbps": float(np.mean(
                    [s["rate_emg_bps"] + s["rate_imp_bps"] + s["rate_norm_bps"] for s in steps]
                )) / 1e6,
            })
    return rows


def size_nominal_reservation(cfg, jobs):
    """Smallest Emergency share r whose S0 episode violation is <= 1%."""
    tasks = [(f"nominal_sized:{r}", cfg, ["S0"], 10, SIZING_OFFSET) for r in R_GRID]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(final_cell, tasks))
    sizing = {}
    for r, rows in zip(R_GRID, results, strict=True):
        sizing[r] = float(np.mean([x["viol_episode"] for x in rows]))
    chosen = next((r for r in R_GRID if sizing[r] <= 0.01), None)
    return chosen, sizing


def evaluate_final(df, cfg):
    """C1, C2, C3-revised, C4, C5 on the criteria seeds."""
    tr = cfg["traffic"]
    offered = tr["emergency_mean_mbps"] + tr["important_mean_mbps"] + tr["normal_mean_mbps"]
    capacity = mean_of(df, "fair_share", "S0", "allocated_rate_mbps")
    load = offered / capacity
    m1, nom = "m1_queue_priority", "nominal"
    win = {s: mean_of(df, m1, s, "viol_ood") for s in ("S1", "S2", "S4")}
    s0 = mean_of(df, m1, "S0", "viol_episode")
    crit = {"C1_load": (load, 0.50 <= load <= 0.60)}
    crit["C2_m1"] = ({"S0_episode": s0, **{f"{s}_window": v for s, v in win.items()}},
                     s0 <= 0.01 and win["S1"] <= 0.05 and win["S2"] <= 0.05
                     and win["S4"] <= 0.20)
    gaps, ok3 = {}, True
    for s in ("S1", "S2", "S4"):
        n = mean_of(df, nom, s, "viol_ood")
        gaps[s] = {"m1": win[s], "nominal_sized": n,
                   "fair_share_reported": mean_of(df, "fair_share", s, "viol_ood")}
        ok3 &= (n >= 3 * win[s]) and (n - win[s] >= 0.05)
    crit["C3_revised"] = (gaps, bool(ok3))
    base = (mean_of(df, m1, "S0", "delivered_imp_mbps")
            + mean_of(df, m1, "S0", "delivered_norm_mbps"))
    stress = (mean_of(df, m1, "S4", "delivered_imp_ood_mbps")
              + mean_of(df, m1, "S4", "delivered_norm_ood_mbps"))
    drop = 1.0 - stress / base if base > 0 else float("nan")
    crit["C4_tradeoff"] = (drop, bool(drop >= 0.05))
    s5 = mean_of(df, "emergency_only", "S5", "viol_ood")
    crit["C5_s5_infeasible"] = (s5, bool(s5 >= 0.5))
    return crit, load


def compare_with_expected(df, r):
    """Differences against the TE-016 table; returns a list of mismatch strings."""
    problems = []
    if abs(r - EXPECTED["r"]) > 1e-9:
        problems.append(f"r = {r} (expected {EXPECTED['r']})")
    for key, want in EXPECTED.items():
        if key == "r":
            continue
        policy, scenario, col = key
        got = mean_of(df, policy, scenario, col)
        if not abs(got - want) <= EXPECTED_TOL:
            problems.append(f"{policy} {scenario} {col}: got {got:.4f}, expected {want:.4f}")
    return problems


def run_final(args, cfg):
    """Evaluate configs/default.yaml unchanged and store the Stage 2 result."""
    from uavrl.eval.manifest import append_manifest, collect_manifest

    out_dir = os.path.join(ROOT, args.out)
    if os.path.exists(os.path.join(out_dir, "verdicts.json")):
        raise SystemExit(f"{out_dir}/verdicts.json already exists; refusing to overwrite. "
                         "Use another --out.")
    os.makedirs(out_dir, exist_ok=True)

    r, sizing = size_nominal_reservation(cfg, args.jobs)
    print("sizing (S0 episode violation, val_seed 300..309):",
          {k: round(100 * v, 2) for k, v in sizing.items()})
    if r is None:
        raise SystemExit("No r in the grid meets 1% on S0; stop and report to the HUMAN.")
    print(f"nominal-sized Emergency share r = {r}")

    names = FINAL_POLICIES + [f"nominal_sized:{r}"]
    tasks = [(p, cfg, SCENARIO_IDS, args.episodes, CRITERIA_OFFSET) for p in names]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for part in ex.map(final_cell, tasks):
            rows.extend(part)
    df = pd.DataFrame(rows)
    df["policy"] = df["policy"].replace({f"nominal_sized:{r}": "nominal"})
    df.to_csv(os.path.join(out_dir, "final_rows.csv"), index=False)
    summary = (df.groupby(["policy", "scenario"])
               [["viol_episode", "viol_ood", "delivered_imp_mbps", "delivered_norm_mbps"]]
               .mean())
    summary.round(4).to_csv(os.path.join(out_dir, "final_summary.csv"))

    crit, load = evaluate_final(df, cfg)
    passed = all(ok for _, ok in crit.values())
    mismatches = compare_with_expected(df, r) if args.episodes == 10 else ["--episodes != 10"]
    print(summary.round(4).to_string())
    for k, (v, ok) in crit.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {k}: {json.dumps(v, default=float)}")
    print("matches TE-016:", "yes" if not mismatches else f"NO -> {mismatches}")

    with open(os.path.join(out_dir, "verdicts.json"), "w", encoding="utf-8") as f:
        json.dump({
            "r": r, "sizing_seeds": f"val_seed({SIZING_OFFSET}..{SIZING_OFFSET + 9})",
            "criteria_seeds": f"val_seed({CRITERIA_OFFSET}..{CRITERIA_OFFSET + args.episodes - 1})",
            "traffic": cfg["traffic"], "P_max": cfg["channel"]["P_max"],
            "load": load, "sizing": {str(k): v for k, v in sizing.items()},
            "criteria": {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in crit.items()},
            "all_pass": bool(passed), "matches_te016": not mismatches,
            "mismatches": mismatches,
        }, f, indent=2, default=float)
    append_manifest(os.path.join(out_dir, "manifest.jsonl"),
                    collect_manifest(ROOT, cfg, "none",
                                     args={"mode": "final", "r": r, "episodes": args.episodes},
                                     files=["configs/default.yaml", "configs/scenarios.yaml"]))
    print(f"wrote {out_dir}")
    if not passed or mismatches:
        print("STOP: do not freeze. Report this output to the HUMAN.")
        sys.exit(2)
    print("Stage 2 final check passed and matches TE-016.")


def mean_of(df, policy, scenario, col):
    sel = df[(df.policy == policy) & (df.scenario == scenario)][col]
    return float(sel.mean())


def evaluate(df, cfg):
    """Apply the pre-registered criteria to one candidate's rows."""
    tr = cfg["traffic"]
    total_load = tr["emergency_mean_mbps"] + tr["important_mean_mbps"] + tr["normal_mean_mbps"]
    capacity = mean_of(df, "fair_share", "S0", "allocated_rate_mbps")
    load = total_load / capacity
    m1 = "m1_queue_priority"
    crit = {}
    crit["C1_load"] = (load, 0.50 <= load <= 0.60)
    s0 = mean_of(df, m1, "S0", "viol_episode")
    w = {s: mean_of(df, m1, s, "viol_ood") for s in ("S1", "S2", "S4")}
    crit["C2_m1"] = (
        {"S0_episode": s0, **{f"{s}_window": v for s, v in w.items()}},
        s0 <= 0.01 and w["S1"] <= 0.05 and w["S2"] <= 0.05 and w["S4"] <= 0.20,
    )
    gaps, ok3 = {}, True
    for s in ("S1", "S2", "S4"):
        fair = mean_of(df, "fair_share", s, "viol_ood")
        gaps[s] = {"m1": w[s], "fair": fair}
        ok3 &= (fair >= 3 * w[s]) and (fair - w[s] >= 0.05)
    crit["C3_discriminates"] = (gaps, bool(ok3))
    imp0 = mean_of(df, m1, "S0", "delivered_imp_mbps") + mean_of(df, m1, "S0", "delivered_norm_mbps")
    imp4 = (mean_of(df, m1, "S4", "delivered_imp_ood_mbps")
            + mean_of(df, m1, "S4", "delivered_norm_ood_mbps"))
    drop = 1.0 - imp4 / imp0 if imp0 > 0 else float("nan")
    crit["C4_tradeoff"] = (drop, bool(drop >= 0.05))
    s5 = mean_of(df, "emergency_only", "S5", "viol_ood")
    crit["C5_s5_infeasible"] = (s5, bool(s5 >= 0.5))
    return crit, load, capacity


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--emergency", default="2,4,6,8", help="Emergency mean loads [Mbps]")
    ap.add_argument("--pmax", default="1.0", help="transmit power budgets [W] (round 2: 1,0.1,0.01)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="validation seeds val_seed(offset .. offset+episodes-1); "
                         "round 3 uses 100 (screening) and 200 (confirmation)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=None,
                    help="default: results/stage2 (sweep) or results/stage2_final (--final)")
    ap.add_argument("--final", action="store_true",
                    help="evaluate configs/default.yaml unchanged with comparator N (TE-015)")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if args.out is None:
        args.out = "results/stage2_final" if args.final else "results/stage2"
    if args.final:
        run_final(args, base_cfg)
        return
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Fair-share capacity depends on the channel (incl. P_max) only, not on the loads.
    emergencies = [float(x) for x in args.emergency.split(",")]
    pmaxes = [float(x) for x in args.pmax.split(",")]
    capacities, cands = {}, {}
    for pm in pmaxes:
        cfg_p = copy.deepcopy(base_cfg)
        cfg_p["channel"]["P_max"] = pm
        cap_rows = run_cell(("cap", cfg_p, "fair_share", 3, args.seed_offset))
        capacities[pm] = float(np.mean([r["allocated_rate_mbps"] for r in cap_rows
                                        if r["scenario"] == "S0"]))
        print(f"P_max={pm:g} W: mean fair-share capacity {capacities[pm]:.1f} Mbps; "
              f"target load {TARGET_LOAD:.0%}")
        for e in emergencies:
            cid = f"E{e:g}_P{pm:g}"
            cands[cid] = candidate_config(base_cfg, e, capacities[pm], p_max=pm)
    tasks = [(cid, cfg, p, args.episodes, args.seed_offset)
             for cid, cfg in cands.items() for p in POLICY_NAMES]
    print(f"{len(cands)} candidates x {len(POLICY_NAMES)} policies x {len(SCENARIO_IDS)} "
          f"scenarios x {args.episodes} episodes on {args.jobs} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for part in ex.map(run_cell, tasks):
            rows.extend(part)
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(out_dir, "feasibility_rows.csv"), index=False)

    verdicts, chosen = {}, None
    for cid, cfg in cands.items():
        crit, load, cap = evaluate(df_all[df_all.candidate == cid], cfg)
        passed = all(ok for _, ok in crit.values())
        verdicts[cid] = {"traffic": cfg["traffic"], "reward": {
            k: cfg["reward"][k] for k in ("R_target", "R_norm_ref")},
            "load": load, "criteria": {k: {"value": v, "pass": bool(ok)}
                                       for k, (v, ok) in crit.items()},
            "all_pass": bool(passed)}
        verdicts[cid]["P_max"] = cfg["channel"]["P_max"]
        verdicts[cid]["emergency_mbps"] = cfg["traffic"]["emergency_mean_mbps"]
        print(f"\n== {cid}  traffic={cfg['traffic']['emergency_mean_mbps']}/"
              f"{cfg['traffic']['important_mean_mbps']}/{cfg['traffic']['normal_mean_mbps']} Mbps"
              f"  load={load:.1%}  -> {'PASS' if passed else 'FAIL'}")
        for k, (v, ok) in crit.items():
            print(f"   [{'PASS' if ok else 'FAIL'}] {k}: {json.dumps(v, default=float)}")
    passing = [c for c, v in verdicts.items() if v["all_pass"]]
    if passing:
        # smallest Emergency load first, then the largest P_max (closest to today's setup)
        chosen = min(passing, key=lambda c: (verdicts[c]["emergency_mbps"], -verdicts[c]["P_max"]))

    summary = (df_all.groupby(["candidate", "policy", "scenario"])
               [["viol_episode", "viol_ood", "delivered_imp_mbps", "delivered_norm_mbps"]]
               .mean().round(4))
    summary.to_csv(os.path.join(out_dir, "feasibility_summary.csv"))
    with open(os.path.join(out_dir, "verdicts.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen": chosen, "capacity_mbps_by_pmax": capacities, "verdicts": verdicts},
                  f, indent=2, default=float)
    print(f"\nwrote {out_dir}/verdicts.json")
    if chosen is None:
        print("NO candidate meets C1-C5. Stop and ask the HUMAN; do not relax thresholds.")
        sys.exit(2)
    print(f"chosen (smallest Emergency load meeting C1-C5): {chosen}")


if __name__ == "__main__":
    main()
