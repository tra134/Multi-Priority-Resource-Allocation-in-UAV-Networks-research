"""S-3 sensitivity analysis runner (docs/lin2025_spec.md Section 11, "S-3 detailed registration").

Runs every registered (variant, policy, N, sic) combination on validation seeds and writes, into a
NEW folder --out (must not exist): per_seed.csv, summary.csv, variants.yaml (the variant configs
used), git_diff.patch and manifest.json. Seeds >= 5000 (test seeds) are refused.

Combinations: the ten variants in configs/s3_variants.yaml x the four policies x N in --n-list, SIC
on, mode POS; plus A8-lit with SIC off for NearestUAVFullPower only. For that one combination the
reference V0 with SIC off (NearestUAVFullPower) is also run, so the paired difference compares
like with like; the A8-lit SIC-off row is also compared with V0 SIC on.

Example (small smoke test, scratch folder):
  python scripts/run_s3.py --out /tmp/s3_smoke --n-seeds 2 --n-list 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_SEED_MIN = 5000
POLICY_ORDER = ["LocalComputing", "NearestUAVFullPower", "RandomPolicy", "GreedyLocalOrOffload"]
METRICS = ("success_rate", "mean_latency_ms", "mean_energy_mw")
N_BOOT = 10000
BOOT_SEED = 0
MIN_PP = 0.01  # |mean success difference| >= 1 percentage point
CLEAR = "changes offloading difficulty"
NOT_CLEAR = "no clear change"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="new output folder (must not exist)")
    p.add_argument("--seeds-start", type=int, default=2000)
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-list", nargs="+", type=int, default=[2, 8, 12])
    p.add_argument("--config", default=str(ROOT / "configs" / "lin2025.yaml"))
    p.add_argument("--variants", default=str(ROOT / "configs" / "s3_variants.yaml"))
    return p


def combos(variant_names: list[str], n_list: list[int]) -> list[tuple[str, str, int, bool]]:
    """All registered (variant, policy, N, sic) combinations, in a fixed order."""
    out = [(v, p, n, True) for v in variant_names for n in n_list for p in POLICY_ORDER]
    for n in n_list:  # A8-lit with SIC off for NearestUAVFullPower only (+ its V0 SIC-off reference)
        out.append(("V0", "NearestUAVFullPower", n, False))
        out.append(("A8-lit", "NearestUAVFullPower", n, False))
    return out


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


def label(mean: float, lo: float, hi: float) -> str:
    """Descriptive label of the registration: CI of the success difference excludes 0 and >= 1 pp."""
    if not math.isnan(mean) and (lo > 0 or hi < 0) and abs(mean) >= MIN_PP:
        return CLEAR
    return NOT_CLEAR


def build_summary(per_seed):
    """Paired difference of every non-V0 combination against V0 (same policy, N, seeds)."""
    import pandas as pd

    rows = []
    key = ["policy", "n_sensors", "seed"]
    refs = per_seed[per_seed["variant"] == "V0"]
    for (variant, policy, n, sic), grp in per_seed.groupby(["variant", "policy", "n_sensors", "sic"], sort=False):
        if variant == "V0":
            continue
        # reference: V0 at the same sic; the A8-lit SIC-off row is also compared with V0 SIC on
        ref_sics = [sic] if sic else [False, True]
        for ref_sic in ref_sics:
            ref = refs[(refs["policy"] == policy) & (refs["n_sensors"] == n) & (refs["sic"] == ref_sic)]
            if ref.empty:
                continue
            m = grp.merge(ref, on=key, suffixes=("", "_v0"))
            row = {
                "variant": variant, "policy": policy, "n_sensors": n, "sic": sic,
                "reference": f"V0 sic={'on' if ref_sic else 'off'}", "n_pairs": len(m),
            }
            for met in METRICS:
                mean, lo, hi, k = paired_bootstrap(m[met] - m[f"{met}_v0"])
                row[f"{met}_variant_mean"] = float(m[met].mean())
                row[f"{met}_v0_mean"] = float(m[f"{met}_v0"].mean())
                row[f"{met}_diff_mean"], row[f"{met}_diff_ci_lo"], row[f"{met}_diff_ci_hi"] = mean, lo, hi
                row[f"{met}_n_finite_pairs"] = k
            row["label"] = label(
                row["success_rate_diff_mean"], row["success_rate_diff_ci_lo"], row["success_rate_diff_ci_hi"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(out: Path, args, seeds, variants, control_ok: bool) -> None:
    import dataclasses

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
        # new code files are untracked until committed and are not in `git diff`
        "untracked_code_files_sha256": {
            f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in untracked
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "n_list": args.n_list,
        "mode": "POS",
        "base_config": args.config,
        "variants": {k: dataclasses.asdict(v) for k, v in variants.items()},
        "n_boot": N_BOOT,
        "bootstrap_rng_seed": BOOT_SEED,
        "label_rule": f"CI of success diff excludes 0 and |mean diff| >= {MIN_PP} -> '{CLEAR}', else '{NOT_CLEAR}'",
        "control_local_computing_identical_to_v0": control_ok,
        "llm_client": "none",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = list(range(args.seeds_start, args.seeds_start + args.n_seeds))
    if args.n_seeds < 1 or any(s >= TEST_SEED_MIN for s in seeds):
        print("refusing: seeds must be validation seeds (< 5000); test seeds are never used in S-3", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists():
        print(f"refusing: {out} already exists (use a new folder)", file=sys.stderr)
        return 2

    import pandas as pd

    from uavmec.baselines.policies import POLICIES
    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.env.s3_variants import load_s3_variants
    from uavmec.eval.runner import run_episode

    variants = load_s3_variants(args.variants)
    out.mkdir(parents=True)
    shutil.copy(args.variants, out / "variants.yaml")
    rows = []
    todo = combos(list(variants), args.n_list)
    for i, (vname, pname, n, sic) in enumerate(todo, 1):
        cfg = load_config(args.config, n_sensors=n)
        env = LinMECEnv(cfg, sic=sic, mode="POS", s3_variant=variants[vname])
        policy = POLICIES[pname]()
        for s in seeds:  # RandomPolicy rng = default_rng(seed) per episode (run_episode); env never uses it
            r = run_episode(env, policy, s)
            r.pop("policy")
            rows.append({"variant": vname, "policy": pname, "n_sensors": n, "sic": sic, **r})
        print(f"[{i}/{len(todo)}] {vname} {pname} N={n} sic={sic}", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)

    # control: LocalComputing must be identical to V0 in every variant (a difference is a bug)
    cols = [c for c in per_seed.columns if c not in ("variant",)]
    loc = per_seed[(per_seed["policy"] == "LocalComputing") & per_seed["sic"]]
    base = loc[loc["variant"] == "V0"].reset_index(drop=True)
    control_ok = all(
        loc[loc["variant"] == v][cols].reset_index(drop=True).equals(base[cols]) for v in variants
    )
    if not control_ok:
        print("ERROR: LocalComputing differs from V0 under some variant (bug)", file=sys.stderr)
    summary = build_summary(per_seed)
    summary.to_csv(out / "summary.csv", index=False)
    write_manifest(out, args, seeds, variants, control_ok)
    return 0 if control_ok else 1


if __name__ == "__main__":
    sys.exit(main())
