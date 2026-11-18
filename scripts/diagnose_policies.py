"""S-5: diagnostics of the learned policies' actions (docs/lin2025_spec.md Section 11, registered 2026-11-18).

Question: do the trained policies behave like NearestUAVFullPower (offload to the nearest UAV at full
power and bandwidth)? Reads <run>/manifest.json and <run>/final.pt (runs written by scripts/train.py),
rebuilds env + learner as scripts/evaluate_learned.py does, runs greedy episodes (explore=False) on
VALIDATION seeds (default 2000-2099; any seed >= 5000 is refused), and records the actions.
NearestUAVFullPower is evaluated on the same seeds and, slot by slot, on the same environment state.
Writes to --out (required, no default): per_run.csv, summary.csv, manifest.json. Fractions are in [0, 1].

Example:
  python scripts/diagnose_policies.py --runs results/s5_inputs_2026-11-18/pos_maddpg_sic_seed0 --out /tmp/s5_scratch
"""

from __future__ import annotations

import argparse
import csv
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

MAX_SEED_EXCLUSIVE = 5000  # spec Section 8: 5000-5999 are test seeds, never used here
N_BOOT = 10_000  # spec S-5
BOOT_SEED = 0
DECILES = tuple(range(10, 100, 10))
# Reading rule, fixed in spec Section 11 (S-5) before any run: descriptive only.
CLOSE_AGREEMENT, CLOSE_POWER, CLOSE_BW = 0.9, 0.9, 0.9
HEUR = "NearestUAVFullPower"


def git_info() -> dict:
    def run(*a):
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_seeds(start: int, n: int) -> None:
    """Refuse test seeds (>= 5000) and empty or negative ranges. Raises ValueError."""
    if n < 1 or start < 0:
        raise ValueError("need --n-seeds >= 1 and --seeds-start >= 0")
    if start + n - 1 >= MAX_SEED_EXCLUSIVE:
        raise ValueError(f"seeds {start}..{start + n - 1} reach test seeds (>= {MAX_SEED_EXCLUSIVE}); refused")


# ------------------------------------------------------------------ pure metric functions (unit-tested)
def _flat(slots):
    return [a for slot in slots for a in slot]


def deciles(values) -> list[float]:
    """10th, 20th, ..., 90th percentiles (linear interpolation); NaNs if there are no values."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return [float("nan")] * len(DECILES)
    return [float(x) for x in np.percentile(v, DECILES)]


def shared_group_fraction(slots) -> float:
    """Fraction of offloaded tasks that share a (UAV, subcarrier) group with another sensor in the same slot.

    slots: list (per slot) of lists of actions (objects with .target, .subcarrier). NaN if nothing is offloaded.
    """
    shared = total = 0
    for slot in slots:
        counts: dict = {}
        for a in slot:
            if a.target > 0:
                counts[(a.target, a.subcarrier)] = counts.get((a.target, a.subcarrier), 0) + 1
        total += sum(counts.values())
        shared += sum(c for c in counts.values() if c >= 2)
    return shared / total if total else float("nan")


def action_metrics(slots, heur_slots, nearest_slots) -> dict:
    """Registered per-run action metrics over all sensor-slots.

    slots / heur_slots: per-slot lists of SensorAction of the learner / of NearestUAVFullPower on the same states.
    nearest_slots: per-slot arrays with the nearest UAV index (0-based) of each sensor.
    """
    acts, heur = _flat(slots), _flat(heur_slots)
    nearest = [int(k) for arr in nearest_slots for k in arr]
    assert len(acts) == len(heur) == len(nearest)
    n = len(acts)
    off = [a for a in acts if a.target > 0]
    nearest_hits = sum(1 for a, k in zip(acts, nearest, strict=True) if a.target == k + 1)
    agree = sum(1 for a, h in zip(acts, heur, strict=True) if a.target == h.target)
    pw, bw = [a.power_frac for a in off], [a.bw_frac for a in off]
    out = {
        "n_sensor_slots": n,
        "n_offloaded": len(off),
        "offload_frac": len(off) / n if n else float("nan"),
        # registered "fraction offloaded to the nearest UAV": over all sensor-slots (a local task is not a hit)
        "nearest_frac": nearest_hits / n if n else float("nan"),
        "nearest_frac_of_offloaded": (sum(1 for a, k in zip(acts, nearest, strict=True)
                                          if a.target > 0 and a.target == k + 1) / len(off)) if off else float("nan"),
        "agreement": agree / n if n else float("nan"),
        "power_mean": float(np.mean(pw)) if pw else float("nan"),
        "bw_mean": float(np.mean(bw)) if bw else float("nan"),
        "shared_group_frac": shared_group_fraction(slots),
        "heur_shared_group_frac": shared_group_fraction(heur_slots),
    }
    for q, v in zip(DECILES, deciles(pw), strict=True):
        out[f"power_d{q}"] = v
    for q, v in zip(DECILES, deciles(bw), strict=True):
        out[f"bw_d{q}"] = v
    return out


def reading_label(m: dict) -> str:
    """Spec S-5 reading rule: close to the heuristic iff agreement >= 0.9 AND mean power >= 0.9 AND mean bw >= 0.9."""
    ok = (m["agreement"] >= CLOSE_AGREEMENT and m["power_mean"] >= CLOSE_POWER and m["bw_mean"] >= CLOSE_BW)
    return "close to the heuristic" if ok else "deviates from the heuristic"  # NaN compares False -> deviates


def paired_difference(learned, heur, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05) -> dict:
    """Paired difference (learned - heuristic) over seeds: mean and percentile bootstrap CI of the mean.

    Pairs where either value is NaN are dropped (n_pairs reports how many remain). Resampling over seeds,
    RNG seed 0, 10 000 resamples (spec S-5).
    """
    a, b = np.asarray(learned, dtype=float), np.asarray(heur, dtype=float)
    d = (a - b)[~(np.isnan(a) | np.isnan(b))]
    if d.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pairs": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    lo, hi = np.quantile(d[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_pairs": int(d.size)}


# ------------------------------------------------------------------ evaluation loop
def run_learner(env, learner, heur_policy, seeds):
    """Greedy episodes. Returns (learner slot actions, heuristic slot actions, nearest-UAV slot arrays, per-seed metrics).

    At every slot the heuristic's action is computed on the same env state (it is deterministic and does not touch
    the env), but only the learner's action is executed. Task streams depend on the seed only (A-36).
    """
    from uavmec.baselines.policies import _nearest_uav
    from uavmec.eval.metrics import summarize_outcomes
    slots, hslots, nslots, rows = [], [], [], []
    rng = np.random.default_rng(0)  # unused by NearestUAVFullPower
    for s in seeds:
        obs, info = env.reset(seed=s)
        state, ep = info["state"], []
        while True:
            acts = learner.act(obs, state, explore=False)
            slots.append(acts)
            hslots.append(heur_policy.act(env, obs, rng))
            nslots.append(_nearest_uav(env))
            obs, _r, term, trunc, info = env.step(acts)
            state = info["state"]
            ep.append(info["outcomes"])
            if term or trunc:
                break
        rows.append({"seed": int(s), **summarize_outcomes(ep)})
    return slots, hslots, nslots, rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories (manifest.json + final.pt)")
    p.add_argument("--out", required=True, help="output directory (required; must not hold per_run.csv yet)")
    p.add_argument("--seeds-start", type=int, default=2000, help="first seed (validation seeds 2000-2999)")
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--env-config", default=str(ROOT / "configs" / "lin2025.yaml"))
    return p


def method_name(algo: str, mode: str, sic: bool) -> str:
    return f"{mode}_{algo.upper()}_{'NOMA' if sic else 'noSIC'}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_seeds(args.seeds_start, args.n_seeds)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    runs = []
    for r in args.runs:
        d = Path(r)
        for f in ("final.pt", "manifest.json"):
            if not (d / f).exists():
                print(f"error: {d} has no {f}", file=sys.stderr)
                return 2
        runs.append(d)
    out = Path(args.out)
    if (out / "per_run.csv").exists():
        print(f"error: {out}/per_run.csv exists; refusing to overwrite (use a new --out)", file=sys.stderr)
        return 2

    import torch

    torch.set_num_threads(2)
    from uavmec.baselines.policies import NearestUAVFullPower
    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.eval.runner import run_episode

    seeds = list(range(args.seeds_start, args.seeds_start + args.n_seeds))
    out.mkdir(parents=True, exist_ok=True)
    heur = NearestUAVFullPower()
    heur_cache: dict = {}  # n_sensors -> per-seed heuristic rows (POS env, SIC on; same streams for every run, A-36)
    rows, run_info = [], []
    for d in runs:
        man = json.loads((d / "manifest.json").read_text())
        ca = man["cli_args"]
        algo, mode, sic, n, seed = ca["algo"], ca["mode"], bool(ca["sic"]), int(ca["n_sensors"]), int(ca["seed"])
        cfg = load_config(args.env_config, n_sensors=n)
        env = LinMECEnv(cfg, sic=sic, mode=mode)
        if n not in heur_cache:
            henv = LinMECEnv(cfg, sic=True, mode="POS")
            heur_cache[n] = [run_episode(henv, heur, s) for s in seeds]
        if algo == "maddpg":
            from uavmec.agents.maddpg import MADDPG as Learner
        else:
            from uavmec.agents.mappo import MAPPO as Learner
        learner = Learner(env.n_agents, env.obs_dim, env.state_dim, cfg.n_uavs, cfg.n_subcarriers,
                          man["train_config"][algo], mode == "FOS", seed)
        pt = d / "final.pt"
        ck = torch.load(pt, weights_only=False)
        learner.load_state_dict(ck["learner"])
        slots, hslots, nslots, ep_rows = run_learner(env, learner, heur, seeds)
        m = action_metrics(slots, hslots, nslots)
        hrows = heur_cache[n]
        succ = paired_difference([r["success_rate"] for r in ep_rows], [r["success_rate"] for r in hrows])
        lat = paired_difference([r["mean_latency_ms"] for r in ep_rows], [r["mean_latency_ms"] for r in hrows])
        name = method_name(algo, mode, sic)
        row = {"run": d.name, "method": name, "n_sensors": n, "seed": seed, **m,
               "success_rate": float(np.mean([r["success_rate"] for r in ep_rows])),
               "heur_success_rate": float(np.mean([r["success_rate"] for r in hrows])),
               "success_diff_pp": 100 * succ["mean"], "success_diff_ci_lo_pp": 100 * succ["ci_lo"],
               "success_diff_ci_hi_pp": 100 * succ["ci_hi"],
               "latency_diff_ms": lat["mean"], "latency_diff_ci_lo_ms": lat["ci_lo"],
               "latency_diff_ci_hi_ms": lat["ci_hi"], "latency_n_pairs": lat["n_pairs"],
               "label": reading_label(m)}
        rows.append(row)
        run_info.append({"run_dir": str(d), "method": name, "seed": seed, "final_pt_sha256": sha256(pt),
                         "run_git_commit": man.get("git_commit", "unknown")})
        print(f"{d.name}: offload={m['offload_frac']:.3f} agree={m['agreement']:.3f} power={m['power_mean']:.3f} "
              f"bw={m['bw_mean']:.3f} succ_diff={row['success_diff_pp']:+.2f}pp [{row['label']}]", flush=True)

    with (out / "per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    num_cols = [k for k, v in rows[0].items() if isinstance(v, float) and k not in ("seed",)]
    srows = []
    for meth in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == meth]
        row = {"method": meth, "n_runs": len(rs),
               "n_close_to_heuristic": sum(r["label"] == "close to the heuristic" for r in rs)}
        for k in num_cols:
            vals = np.array([r[k] for r in rs], dtype=float)
            row[f"{k}_mean"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
            row[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if np.sum(~np.isnan(vals)) > 1 else 0.0
        srows.append(row)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)

    manifest = {**git_info(), "timestamp": datetime.now(timezone.utc).isoformat(), "cli_args": vars(args),
                "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "greedy": True, "env_version": "v1.0 (default, no s3_variant)",
                "reading_rule": {"agreement>=": CLOSE_AGREEMENT, "power_mean>=": CLOSE_POWER, "bw_mean>=": CLOSE_BW},
                "bootstrap": {"n_boot": N_BOOT, "rng_seed": BOOT_SEED, "alpha": 0.05},
                "runs": run_info, "llm_client": "none",
                "torch": torch.__version__, "numpy": np.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if any(math.isnan(r["agreement"]) for r in rows):
        print("warning: NaN agreement in some run", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
