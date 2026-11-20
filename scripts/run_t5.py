"""Test T-5 (docs/lin2025_spec.md Section 13): Table 4 slot replay in LinMECEnv versus extra link loss L.

Replays the Table 4 slot (N = 8, K = 2, SIC, POS, current c_cyc and kappa) under readings R1
(every offloader -> UAV1 at (30, 53)) and R2 (UAV1 serves S0, S4; UAV0 at (77, 87) serves S3, S5,
S6, S7) for L from -10 to +20 dB in 0.25 dB steps (A-44: gain scaled by 10^(-L/10)). Feasible set
F = {L : 8/8 under R1 and R2}; decision rule per Section 13. Also a descriptive path-loss exponent
scan (never decides). Writes a NEW folder --out (must not exist). No learning, no LLM.

Example (scratch): python scripts/run_t5.py --out /tmp/t5_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG = "configs/lin2025.yaml"
GRID = [x / 4 for x in range(-40, 81)]  # -10 .. +20 dB, 0.25 dB steps
ALPHAS = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
UAV_POS = np.array([[77.0, 87.0], [30.0, 53.0]])  # Table 3: uav0, uav1
# Table 4: id, (x, y), size (Mbit), deadline (s), action (target 0 = local, else UAV index 1 or 2;
# carrier, bw %, power %), paper success. Target here is the R1 (literal) UAV: offloaders -> UAV1.
SENSORS = [
    ("MU0", (35, 14), 1.1, 0.042, (1, 0, 99.9, 99.9), True),
    ("MU1", (2, 21), 0.825, 0.033, None, True),
    ("MU2", (55, 35), 0.9, 0.038, None, True),
    ("MU3", (82, 21), 0.9, 0.025, (1, 2, 47.8, 36.2), True),
    ("MU4", (62, 35), 0.9, 0.035, (1, 3, 61.1, 31.1), True),
    ("MU5", (1, 59), 1.31, 0.033, (1, 2, 45.3, 35.7), True),
    ("MU6", (56, 40), 1.11, 0.021, (1, 1, 30.8, 51.0), False),
    ("MU7", (46, 34), 0.51, 0.014, (1, 1, 42.2, 49.0), True),
]
R2_UAV0 = {"MU3", "MU5", "MU6", "MU7"}  # R2: these are served by UAV0 (target 1), others UAV1


def table4_gain(loss_db: float, alpha: float = 2.0) -> np.ndarray:
    """(8, 2) power gain of the Table 4 sensors to UAV0, UAV1: beta0 10^(-L/10) / (H^2 + d^2)^(alpha/2).

    alpha = 2 is the A-1 law. loss_db is the A-44 extra link loss.
    """
    from uavmec.config import load_config

    cfg = load_config(ROOT / CONFIG)
    pos = np.array([s[1] for s in SENSORS], dtype=float)
    d2 = np.sum((pos[:, None, :] - UAV_POS[None, :, :]) ** 2, axis=-1)
    return cfg.beta0 * 10.0 ** (-loss_db / 10.0) / (cfg.uav_height_m**2 + d2) ** (alpha / 2.0)


def replay(reading: str, loss_db: float = 0.0, alpha: float = 2.0, sic: bool = True) -> list[dict]:
    """One LinMECEnv step on the Table 4 slot. Returns one dict per sensor."""
    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.env.types import SensorAction

    if reading not in ("R1", "R2"):
        raise ValueError(reading)
    cfg = load_config(ROOT / CONFIG, n_sensors=8)
    env = LinMECEnv(cfg, sic=sic, mode="POS")
    env.reset(seed=0)
    env.sensor_pos = np.array([s[1] for s in SENSORS], dtype=float)
    env.uav_pos = UAV_POS.copy()
    env.task_sizes = np.array([s[2] * 1e6 for s in SENSORS])
    env.task_deadlines = np.array([s[3] for s in SENSORS])
    env.battery = np.full(8, cfg.sensor_battery_j[1])
    env.channel_gain = table4_gain(loss_db, alpha)  # A-44 (and descriptive alpha)
    actions = []
    for name, _pos, _c, _d, act, _ok in SENSORS:
        if act is None:
            actions.append(SensorAction(target=0))
            continue
        target = 1 if (reading == "R2" and name in R2_UAV0) else 2
        _, carrier, bw, p = act
        actions.append(SensorAction(target=target, subcarrier=carrier, power_frac=p / 100, bw_frac=bw / 100))
    _, _, _, _, info = env.step(actions)
    rows = []
    for s, o in zip(SENSORS, info["outcomes"], strict=True):
        rows.append({
            "sensor": s[0], "mode": o.mode,
            "rate_mbps": None if o.rate_bps is None else round(o.rate_bps / 1e6, 2),
            "sinr": None if o.sinr is None else round(o.sinr, 3),
            "latency_ms": round(o.latency_s * 1e3, 2) if math.isfinite(o.latency_s) else float("inf"),
            "deadline_ms": round(s[3] * 1e3, 2), "sim_success": bool(o.success),
            "paper_success": s[5], "match": bool(o.success) == s[5], "fail_reason": o.fail_reason or "",
        })
    return rows


def score(rows: list[dict]) -> int:
    return sum(1 for r in rows if r["match"])


def feasible(grid, scores_r1: dict, scores_r2: dict) -> list[float]:
    """L values (in grid order) with 8/8 under both readings."""
    return [L for L in grid if scores_r1.get(L) == 8 and scores_r2.get(L) == 8]


def decide(f: list[float], grid, step: float = 0.25) -> dict:
    """Decision rule of Section 13. Ties of the 0.5 dB rounding go up."""
    if 0.0 in f:
        return {"adopted": False, "reason": "0 in F: no change", "L_star": None, "interval": None}
    if not f:
        return {"adopted": False, "reason": "F empty: no change", "L_star": None, "interval": None}
    runs, cur = [], [f[0]]
    for a, b in zip(f, f[1:], strict=False):
        if abs((b - a) - step) < 1e-9:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    best = max(runs, key=len)  # first of the longest runs on ties (lowest L)
    mid = (best[0] + best[-1]) / 2.0
    l_star = math.floor(mid * 2.0 + 0.5) / 2.0
    return {"adopted": True, "reason": "F non-empty and excludes 0", "L_star": l_star,
            "interval": [best[0], best[-1]], "midpoint": mid, "n_runs": len(runs)}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="new output folder (must not exist)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refusing: {out} already exists", file=sys.stderr)
        return 2
    import pandas as pd

    z = {r: replay(r, 0.0) for r in ("R1", "R2")}
    ref = ROOT / "results" / "T2_offload_physics_2026-11-13" / "table4_replay.csv"
    t2 = pd.read_csv(ref)
    t2 = t2[t2["cand"] == "a_dist_percarrier"]
    check = all(
        bool(e["sim_success"]) == r["sim_success"] and bool(e["match"]) == r["match"]
        for rd in ("R1", "R2")
        for r, (_, e) in zip(z[rd], t2[t2["reading"] == rd].iterrows(), strict=True)
    )
    if not check or score(z["R1"]) + score(z["R2"]) != 14:
        print("STOP: L = 0 replay does not reproduce T-2 (14/16 and outcomes)", file=sys.stderr)
        return 3
    out.mkdir(parents=True)
    detail, s1, s2 = [], {}, {}
    for L in GRID:
        for rd in ("R1", "R2"):
            rows = replay(rd, L)
            (s1 if rd == "R1" else s2)[L] = score(rows)
            detail += [{"L_db": L, "reading": rd, **r} for r in rows]
    pd.DataFrame(detail).to_csv(out / "t5_replay_per_L.csv", index=False)
    pd.DataFrame({"L_db": GRID, "score_R1": [s1[L] for L in GRID], "score_R2": [s2[L] for L in GRID]}
                 ).to_csv(out / "t5_scores.csv", index=False)
    f = feasible(GRID, s1, s2)
    decision = decide(f, GRID)
    arows = []
    for a in ALPHAS:
        sc = {rd: score(replay(rd, 0.0, alpha=a)) for rd in ("R1", "R2")}
        arows.append({"alpha": a, "score_R1": sc["R1"], "score_R2": sc["R2"]})
    pd.DataFrame(arows).to_csv(out / "t5_alpha_descriptive.csv", index=False)
    (out / "t5_decision.json").write_text(json.dumps(
        {"feasible_set": f, "n_feasible": len(f), "L_zero_in_F": 0.0 in f, **decision,
         "t2_check_14_of_16_reproduced": True}, indent=2))
    try:
        commit, dirty = _git("rev-parse", "HEAD").strip(), bool(_git("status", "--porcelain",
                                                                 "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    (out / "manifest.json").write_text(json.dumps({
        "test": "T-5", "git_commit": commit, "git_dirty": dirty, "config": CONFIG,
        "config_sha256": hashlib.sha256((ROOT / CONFIG).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grid_db": [GRID[0], GRID[-1], 0.25], "alphas": ALPHAS, "seeds": "none (deterministic replay)",
        "timestamp": datetime.now(timezone.utc).isoformat(), "llm_client": "none",
    }, indent=2))
    print(json.dumps({"feasible": [f[0], f[-1]] if f else None, **decision}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
