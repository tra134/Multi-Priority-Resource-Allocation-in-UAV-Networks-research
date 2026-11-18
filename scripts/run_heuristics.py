"""Non-learning policies on lin-env-v1.0 and lin-env-v1.1 (docs/lin2025_spec.md Section 12, check 7).

Runs the four non-learning policies on both environments at N in --n-list, SIC on, mode POS,
validation seeds only (seeds >= 5000 are refused). Writes into a NEW folder --out (must not
exist): per_seed.csv, summary.csv (v1.0 mean, v1.1 mean, paired v1.1 - v1.0 difference with a 95%
bootstrap CI, 10000 resamples, RNG seed 0) and manifest.json. Descriptive only; no LLM.

Example (small smoke test, scratch folder):
  python scripts/run_heuristics.py --out /tmp/heur_smoke --n-seeds 3 --n-list 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_SEED_MIN = 5000
POLICY_ORDER = ["LocalComputing", "NearestUAVFullPower", "RandomPolicy", "GreedyLocalOrOffload"]
ENVS = {"v1.0": "configs/lin2025.yaml", "v1.1": "configs/lin2025_v1_1.yaml"}
METRICS = ("success_rate", "mean_latency_ms", "mean_energy_mw")
N_BOOT = 10000
BOOT_SEED = 0


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="new output folder (must not exist)")
    p.add_argument("--seeds-start", type=int, default=2000)
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-list", nargs="+", type=int, default=[2, 8, 12])
    return p


def paired_bootstrap(diff, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Mean and 95% percentile bootstrap CI of paired differences (NaNs dropped), RNG seed 0."""
    import numpy as np

    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi), int(d.size)


def build_summary(per_seed):
    import pandas as pd

    rows = []
    for (policy, n), grp in per_seed.groupby(["policy", "n_sensors"], sort=False):
        a = grp[grp["env"] == "v1.0"].set_index("seed")
        b = grp[grp["env"] == "v1.1"].set_index("seed")
        a, b = a.loc[sorted(a.index)], b.loc[sorted(a.index)]
        row = {"policy": policy, "n_sensors": n, "n_pairs": len(a)}
        for met in METRICS:
            mean, lo, hi, k = paired_bootstrap((b[met] - a[met]).to_numpy())
            row[f"{met}_v1_0_mean"], row[f"{met}_v1_1_mean"] = float(a[met].mean()), float(b[met].mean())
            row[f"{met}_diff_mean"], row[f"{met}_diff_ci_lo"], row[f"{met}_diff_ci_hi"] = mean, lo, hi
            row[f"{met}_n_finite_pairs"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(out: Path, args, seeds) -> None:
    try:
        commit = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--porcelain")
        diff = _git("diff", "HEAD")
        untracked = [
            f for f in _git("ls-files", "--others", "--exclude-standard").split("\n")
            if f.startswith(("src/", "scripts/", "tests/", "configs/"))
        ]
    except (OSError, subprocess.CalledProcessError):
        commit, status, diff, untracked = "unknown", "", "", []
    (out / "git_diff.patch").write_text(diff)
    manifest = {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_diff_patch": "git_diff.patch",
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "untracked_code_files_sha256": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in untracked
        },
        "env_configs": {k: v for k, v in ENVS.items()},
        "env_config_sha256": {k: hashlib.sha256((ROOT / v).read_bytes()).hexdigest() for k, v in ENVS.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds, "n_list": args.n_list, "policies": POLICY_ORDER, "mode": "POS", "sic": True,
        "n_boot": N_BOOT, "bootstrap_rng_seed": BOOT_SEED, "cli_args": vars(args),
        "llm_client": "none",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(range(args.seeds_start, args.seeds_start + args.n_seeds))
    if args.n_seeds < 1 or any(s >= TEST_SEED_MIN for s in seeds):
        print("refusing: seeds must be validation seeds (< 5000); test seeds are never used here", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists():
        print(f"refusing: {out} already exists (use a new folder)", file=sys.stderr)
        return 2

    import pandas as pd

    from uavmec.baselines.policies import POLICIES
    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.eval.runner import run_episode

    out.mkdir(parents=True)
    rows = []
    for env_name, cfg_path in ENVS.items():
        for n in args.n_list:
            env = LinMECEnv(load_config(ROOT / cfg_path, n_sensors=n), sic=True, mode="POS")
            for pname in POLICY_ORDER:
                policy = POLICIES[pname]()
                for s in seeds:
                    r = run_episode(env, policy, s)
                    r.pop("policy")
                    rows.append({"env": env_name, "policy": pname, "n_sensors": n, **r})
                print(f"{env_name} N={n} {pname}", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    build_summary(per_seed).to_csv(out / "summary.csv", index=False)
    write_manifest(out, args, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
