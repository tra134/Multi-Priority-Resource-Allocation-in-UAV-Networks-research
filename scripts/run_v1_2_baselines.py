"""Non-learning baselines on lin-env-v1.0 and lin-env-v1.2 (docs/lin2025_spec.md Section 13).

Policies LocalComputing, RandomPolicy, NearestUAVFullPower, GreedyLocalOrOffload at N in --n-list
(SIC, POS), plus NearestUAVFullPower with noSIC at N = 8 (label NearestUAVFullPower_noSIC).
Validation seeds only (seeds >= 5000 are refused). Writes a NEW folder --out (must not exist):
per_seed.csv, summary.csv (v1.0 mean, v1.2 mean, paired v1.2 - v1.0 difference with a 95% bootstrap
CI, 10000 resamples, RNG seed 0) and manifest.json. Descriptive only; no LLM.

Example (scratch): python scripts/run_v1_2_baselines.py --out /tmp/b_smoke --n-seeds 3 --n-list 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("run_heuristics", ROOT / "scripts" / "run_heuristics.py")
_rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rh)

TEST_SEED_MIN = 5000
POLICY_ORDER = ["LocalComputing", "RandomPolicy", "NearestUAVFullPower", "GreedyLocalOrOffload"]
ENVS = {"v1.0": "configs/lin2025.yaml", "v1.2": "configs/lin2025_v1_2.yaml"}
METRICS = _rh.METRICS
N_BOOT, BOOT_SEED = _rh.N_BOOT, _rh.BOOT_SEED
NOSIC_N = 8
NOSIC_LABEL = "NearestUAVFullPower_noSIC"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="new output folder (must not exist)")
    p.add_argument("--seeds-start", type=int, default=2000)
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-list", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    return p


def build_summary(per_seed):
    import pandas as pd

    rows = []
    for (policy, n), grp in per_seed.groupby(["policy", "n_sensors"], sort=False):
        a = grp[grp["env"] == "v1.0"].set_index("seed")
        b = grp[grp["env"] == "v1.2"].set_index("seed")
        a, b = a.loc[sorted(a.index)], b.loc[sorted(a.index)]
        row = {"policy": policy, "n_sensors": n, "n_pairs": len(a)}
        for met in METRICS:
            mean, lo, hi, k = _rh.paired_bootstrap((b[met] - a[met]).to_numpy(), N_BOOT, BOOT_SEED)
            row[f"{met}_v1_0_mean"], row[f"{met}_v1_2_mean"] = float(a[met].mean()), float(b[met].mean())
            row[f"{met}_diff_mean"], row[f"{met}_diff_ci_lo"], row[f"{met}_diff_ci_hi"] = mean, lo, hi
            row[f"{met}_n_finite_pairs"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def write_manifest(out: Path, args, seeds) -> None:
    try:
        commit = _git("rev-parse", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    (out / "manifest.json").write_text(json.dumps({
        "git_commit": commit, "git_dirty": dirty,
        "env_configs": ENVS,
        "env_config_sha256": {k: hashlib.sha256((ROOT / v).read_bytes()).hexdigest() for k, v in ENVS.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "n_list": args.n_list,
        "policies": POLICY_ORDER + [NOSIC_LABEL], "mode": "POS", "sic": True,
        "nosic_n": NOSIC_N, "n_boot": N_BOOT, "bootstrap_rng_seed": BOOT_SEED,
        "cli_args": vars(args), "llm_client": "none",
    }, indent=2, default=str))


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
            cfg = load_config(ROOT / cfg_path, n_sensors=n)
            jobs = [(p, p, True) for p in POLICY_ORDER]
            if n == NOSIC_N:
                jobs.append(("NearestUAVFullPower", NOSIC_LABEL, False))
            for pname, label, sic in jobs:
                env = LinMECEnv(cfg, sic=sic, mode="POS")
                policy = POLICIES[pname]()
                for s in seeds:
                    r = run_episode(env, policy, s)
                    r.pop("policy")
                    rows.append({"env": env_name, "policy": label, "n_sensors": n, **r})
                print(f"{env_name} N={n} {label}", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    build_summary(per_seed).to_csv(out / "summary.csv", index=False)
    write_manifest(out, args, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
