"""Stage 3 pilot report: do the pilot checkpoints learn, and do they degrade under stress?

Evaluates the pilot SAC checkpoints and three references on VALIDATION seeds only
(``val_seed(offset + episode)``; never test seeds, CLAUDE.md Rule 4), with the
deterministic actor, on S0 (nominal), S1 (surge), S2 (shadow) and S4 (compound).

Policies
--------
sac_n2      Pure SAC, plain N=2 checkpoint
sac_n10     REDQ-SAC, plain N=10 checkpoint
fair_share  equal shares
m1          queue-aware strict priority
nominal     static policy sized for nominal traffic (Emergency share --n-share)

Gates (fixed before any training; docs/agent_guide_stage2_3.md)
---------------------------------------------------------------
G1 Learning        mean reward on S0 is higher than fair share's
G2 Service         on S0, delivered Important >= 50% and Normal >= 50% of the offered load
G3 Nominal safety  S0 episode violation <= 5%
G4 Degradation     in S1 or S2 (reported separately), in-window violation is
                   >= 3x the S0 episode rate AND >= 5 points higher

G1-G3 are required for every SAC checkpoint that is present.  G4 is the plan's gate
and is judged for Pure SAC (sac_n2); it is also reported for sac_n10.
``stage3_pass`` = G1-G3 for all present checkpoints and G4 for sac_n2 in S1 or S2.

Usage
-----
    python scripts/pilot_report.py --ckpt-dir /content/drive/MyDrive/uavrl/ckpt_pilot \\
        --seed 0 --out /content/drive/MyDrive/uavrl/results/stage3_pilot

Output: pilot_report.csv, pilot_report.json, manifest.jsonl in --out.  Refuses to
overwrite an existing pilot_report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_grid import checkpoint_path  # noqa: E402

from uavrl.agents.sac import SACAgent  # noqa: E402
from uavrl.baselines.static_policies import (  # noqa: E402
    QueueAwarePriority,
    fair_share,
    shares_to_action,
)
from uavrl.env.scenarios import make_scenario  # noqa: E402
from uavrl.env.uav_env import UAVEnv  # noqa: E402
from uavrl.eval.manifest import append_manifest, collect_manifest  # noqa: E402
from uavrl.eval.metrics import episode_metrics  # noqa: E402
from uavrl.eval.runner import run_episode  # noqa: E402
from uavrl.seeds import MAX_EPISODES, val_seed  # noqa: E402

SCENARIOS = ["S0", "S1", "S2", "S4"]
SAC_POLICIES = {"sac_n2": 2, "sac_n10": 10}
REFERENCES = ["fair_share", "m1", "nominal"]
METRIC_COLS = [
    "sos_violation_episode", "sos_violation_ood", "delivered_imp_mbps",
    "delivered_norm_mbps", "delivered_imp_ood_mbps", "delivered_norm_ood_mbps",
    "mean_b_share_emg", "mean_p_share_emg", "mean_r_emg", "mean_r_imp", "mean_r_norm",
    "mean_penalty", "mean_reward", "mean_U",
]


def nominal_sized_policy(r):
    rest = 1.0 - r
    shares = [r, rest * 35.0 / 63.0, rest * 28.0 / 63.0]
    action = shares_to_action(shares, shares)
    return lambda obs, info: action


def load_agent(ckpt_dir, seed, n_critics):
    """Return a CPU agent, or None when the checkpoint is missing."""
    path = checkpoint_path(ckpt_dir, seed, n_critics, "")
    if not os.path.exists(path):
        return None, path
    agent = SACAgent(obs_dim=17, act_dim=6, device="cpu", seed=seed,
                     N_critics=n_critics, rng=np.random.default_rng(seed))
    agent.load(path)
    return agent, path


def run_cell(policy_name, policy, scenario_id, cfg, offset, episodes, agent=None):
    """Mean metrics of one (policy, scenario); also mean ensemble uncertainty U_t."""
    slot_ms = float(cfg["timing"]["slot_ms"])
    rows = []
    for ep in range(episodes):
        seed = val_seed(offset + ep)
        env = UAVEnv(config=cfg, scenario=make_scenario(scenario_id, seed), seed=seed)
        us = []
        if agent is not None:
            def act(obs, info, _a=agent, _u=us):
                a = _a.select_action(obs, deterministic=True)
                _u.append(_a.uncertainty(obs, a)[0])
                return a
        else:
            act = policy
        steps, summary = run_episode(env, act, seed=seed)
        m = episode_metrics(steps, slot_ms)
        m["mean_reward"] = summary["mean_reward"]
        m["mean_U"] = float(np.mean(us)) if us else math.nan
        rows.append(m)
    out = {"policy": policy_name, "scenario": scenario_id, "episodes": episodes}
    for col in METRIC_COLS:
        vals = np.array([r.get(col, math.nan) for r in rows], dtype=float)
        out[col] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
    return out


def _get(df, policy, scenario, col):
    sel = df[(df.policy == policy) & (df.scenario == scenario)][col]
    return float(sel.iloc[0]) if len(sel) else math.nan


def compute_gates(df, cfg, present, n_share):
    """Apply G1-G4 to the result table.  ``present`` lists the SAC policies found."""
    tr = cfg["traffic"]
    offered_imp, offered_norm = tr["important_mean_mbps"], tr["normal_mean_mbps"]
    gates = {}
    for p in present:
        g1 = _get(df, p, "S0", "mean_reward") > _get(df, "fair_share", "S0", "mean_reward")
        imp = _get(df, p, "S0", "delivered_imp_mbps")
        norm = _get(df, p, "S0", "delivered_norm_mbps")
        g2 = imp >= 0.5 * offered_imp and norm >= 0.5 * offered_norm
        s0 = _get(df, p, "S0", "sos_violation_episode")
        g3 = s0 <= 0.05
        g4 = {}
        for s in ("S1", "S2"):
            win = _get(df, p, s, "sos_violation_ood")
            g4[s] = {"window": win, "s0_episode": s0,
                     "pass": bool(win >= 3 * s0 and win - s0 >= 0.05)}
        gates[p] = {
            "G1_learning": {"sac_reward": _get(df, p, "S0", "mean_reward"),
                            "fair_share_reward": _get(df, "fair_share", "S0", "mean_reward"),
                            "pass": bool(g1)},
            "G2_service": {"delivered_imp_mbps": imp, "delivered_norm_mbps": norm,
                           "offered_imp_mbps": offered_imp, "offered_norm_mbps": offered_norm,
                           "pass": bool(g2)},
            "G3_nominal_safety": {"s0_episode_violation": s0, "pass": bool(g3)},
            "G4_degradation": {**g4, "pass": bool(g4["S1"]["pass"] or g4["S2"]["pass"])},
            "mean_emergency_share_s0": {
                "bandwidth": _get(df, p, "S0", "mean_b_share_emg"),
                "power": _get(df, p, "S0", "mean_p_share_emg"),
                "nominal_policy_r": n_share,
                "m1_bandwidth": _get(df, "m1", "S0", "mean_b_share_emg"),
            },
            "mean_U_t": {s: _get(df, p, s, "mean_U") for s in SCENARIOS},
        }
    ok_123 = all(gates[p][g]["pass"] for p in present
                 for g in ("G1_learning", "G2_service", "G3_nominal_safety"))
    g4_ref = gates["sac_n2"]["G4_degradation"]["pass"] if "sac_n2" in gates else False
    return gates, bool(ok_123 and g4_ref)


def advice(gates, present):
    """Plain-language reading of the gates, following the table in the guide."""
    if "sac_n2" not in gates:
        return "Pure SAC (sac_n2) checkpoint missing: G4 cannot be judged."
    notes = []
    for p in present:
        for g in ("G1_learning", "G2_service", "G3_nominal_safety"):
            if not gates[p][g]["pass"]:
                notes.append(f"{p} fails {g}: training or reward problem; diagnose with the "
                             "logged reward terms (one retune allowed).")
    g4 = gates["sac_n2"]["G4_degradation"]
    if g4["S1"]["pass"]:
        notes.append("G4 passes for S1: Stage 3 degradation gate met.")
    elif g4["S2"]["pass"]:
        notes.append("G4 passes for S2 but not S1: SAC may be over-reserving for Emergency; "
                     "compare its Emergency share with the nominal policy's r.")
    else:
        notes.append("G4 fails for both S1 and S2: use the one allowed retune, then, if it "
                     "still fails, stop and tell the HUMAN.")
    return " ".join(notes)


def run(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3 pilot report (validation seeds only)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt-dir", required=True,
                    help="folder with sac_seed<k>_N<n>/ run folders or flat checkpoints")
    ap.add_argument("--seed", type=int, default=0, help="training seed of the pilot")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=500,
                    help="validation seeds val_seed(offset .. offset+episodes-1); "
                         "use 600 after a reward retune")
    ap.add_argument("--n-share", type=float, default=0.10,
                    help="Emergency share r of the nominal-sized comparator (Stage 2: 0.10)")
    ap.add_argument("--slots", type=int, default=None, help="episode length (tests only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if not (1 <= args.episodes <= MAX_EPISODES):
        raise SystemExit(f"--episodes must be in [1, {MAX_EPISODES}]")
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.slots:
        cfg["timing"]["episode_slots"] = int(args.slots)

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if os.path.exists(os.path.join(out_dir, "pilot_report.json")):
        raise SystemExit(f"{out_dir}/pilot_report.json exists; refusing to overwrite. "
                         "Use another --out.")
    os.makedirs(out_dir, exist_ok=True)

    agents, found = {}, {}
    for name, n in SAC_POLICIES.items():
        agent, path = load_agent(args.ckpt_dir, args.seed, n)
        found[name] = path if agent is not None else None
        if agent is not None:
            agents[name] = agent
        else:
            print(f"[missing] {name}: no checkpoint at {path}")
    if not agents:
        raise SystemExit("No SAC checkpoint found; nothing to report.")

    refs = {"fair_share": fair_share, "m1": QueueAwarePriority.from_config(cfg),
            "nominal": nominal_sized_policy(args.n_share)}
    rows = []
    for sid in SCENARIOS:
        for name, agent in agents.items():
            print(f"{name} {sid} ...", flush=True)
            rows.append(run_cell(name, None, sid, cfg, args.seed_offset, args.episodes, agent))
        for name in REFERENCES:
            print(f"{name} {sid} ...", flush=True)
            rows.append(run_cell(name, refs[name], sid, cfg, args.seed_offset, args.episodes))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "pilot_report.csv"), index=False)

    present = list(agents)
    gates, passed = compute_gates(df, cfg, present, args.n_share)
    verdict = {
        "stage3_pass": passed,
        "advice": advice(gates, present),
        "checkpoints": found,
        "seeds": f"val_seed({args.seed_offset}..{args.seed_offset + args.episodes - 1})",
        "episodes": args.episodes,
        "env_version": cfg.get("meta", {}).get("env_version"),
        "gates": gates,
    }
    with open(os.path.join(out_dir, "pilot_report.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, default=float)
    append_manifest(os.path.join(out_dir, "manifest.jsonl"),
                    collect_manifest(ROOT, cfg, "none",
                                     args={k: v for k, v in vars(args).items()},
                                     files=["configs/default.yaml", "configs/scenarios.yaml"]))

    pd.set_option("display.width", 200)
    print(df[["policy", "scenario", "sos_violation_episode", "sos_violation_ood",
              "mean_b_share_emg", "mean_reward"]].round(4).to_string(index=False))
    for p in present:
        g = gates[p]
        print(f"{p}: G1 {g['G1_learning']['pass']}  G2 {g['G2_service']['pass']}  "
              f"G3 {g['G3_nominal_safety']['pass']}  G4 S1 {g['G4_degradation']['S1']['pass']} "
              f"S2 {g['G4_degradation']['S2']['pass']}")
    print("stage3_pass:", passed)
    print("advice:", verdict["advice"])
    print("wrote", out_dir)
    return verdict


if __name__ == "__main__":
    run()
