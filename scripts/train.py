"""Train MADDPG or MAPPO on LinMECEnv (Lin et al. 2025 reproduction, docs/lin2025_spec.md Section 10).

A-23 hyperparameters must be approved by the HUMAN before a real training run.
Writes to --out: train_log.csv, eval_log.csv, manifest.json, ckpt.pt, final.pt.
--out is required (no default). Training reset seeds: 10_000 + seed * 1_000_000 + episode.
Periodic evaluation uses validation seeds 2000.. with explore=False (never test seeds).
reward_sum = sum over the T slots of the mean-over-sensors reward (comparable to Fig. 4a).

Examples:
  python scripts/train.py --algo maddpg --mode POS --seed 0 --out /tmp/run_pos_maddpg --resume
  python scripts/train.py --algo mappo --mode POS --no-sic --seed 1 --out /tmp/run_mappo_nosic
  python scripts/train.py --algo maddpg --n-sensors 2 --episodes 3 --set eval_episodes=1 --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VAL_START = 2000
TRAIN_COLS = ["episode", "reward_sum", "success_rate", "mean_latency_ms", "mean_energy_mw", "losses"]
EVAL_COLS = ["episode", "success_rate", "mean_latency_ms", "mean_energy_mw", "n_episodes"]


def git_info() -> dict:
    def run(*a):
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["maddpg", "mappo"], required=True)
    p.add_argument("--mode", choices=["POS", "FOS"], default="POS")
    p.add_argument("--sic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-sensors", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="training seed index")
    p.add_argument("--episodes", type=int, default=None, help="override the config value")
    p.add_argument("--train-config", default=str(ROOT / "configs" / "train_lin2025.yaml"))
    p.add_argument("--env-config", default=str(ROOT / "configs" / "lin2025.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a top-level key or an algo key (YAML value), repeatable")
    p.add_argument("--out", required=True, help="output directory (required)")
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if it exists")
    return p


def read_rows(path: Path, keep_below: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [r for r in csv.DictReader(f) if int(r["episode"]) < keep_below]


def write_rows(path: Path, cols: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def evaluate(env, learner, n_eps: int) -> dict:
    """Greedy (explore=False) evaluation on validation seeds, same metrics as the baselines."""
    from uavmec.eval.metrics import aggregate_episodes, summarize_outcomes

    rows = []
    for s in range(VAL_START, VAL_START + n_eps):
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
    return {k: agg[k] for k in ("success_rate", "mean_latency_ms", "mean_energy_mw", "n_episodes")}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import numpy as np
    import torch
    import yaml

    from uavmec.config import load_config
    from uavmec.env.lin_env import LinMECEnv
    from uavmec.eval.metrics import summarize_outcomes

    torch.set_num_threads(int(os.environ.get("UAVMEC_TORCH_THREADS", "2")))
    tcfg = yaml.safe_load(Path(args.train_config).read_text())
    for kv in args.set:
        key, _, val = kv.partition("=")
        target = tcfg[args.algo] if key in tcfg[args.algo] else tcfg
        target[key] = yaml.safe_load(val)
    if args.episodes is not None:
        tcfg["episodes"] = args.episodes
    cfg = load_config(args.env_config, n_sensors=args.n_sensors)
    fos = args.mode == "FOS"
    env = LinMECEnv(cfg, sic=args.sic, mode=args.mode)
    if args.algo == "maddpg":
        from uavmec.agents.maddpg import MADDPG as Learner
    else:
        from uavmec.agents.mappo import MAPPO as Learner
    learner = Learner(env.n_agents, env.obs_dim, env.state_dim, cfg.n_uavs, cfg.n_subcarriers,
                      tcfg[args.algo], fos, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path, start = out / "ckpt.pt", 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        learner.load_state_dict(ck["learner"])
        start = ck["episode"]
    train_rows = read_rows(out / "train_log.csv", start) if start else []
    eval_rows = read_rows(out / "eval_log.csv", start + 1) if start else []
    manifest = {
        **git_info(), "timestamp": datetime.now(timezone.utc).isoformat(), "cli_args": vars(args),
        "train_config": tcfg, "env_config": dataclasses.asdict(cfg), "training_seed_index": args.seed,
        "reset_seed_formula": "10000 + seed * 1000000 + episode", "eval_seeds_start": VAL_START,
        "resumed_from_episode": start, "llm_client": "none",
        "torch": torch.__version__, "numpy": np.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    def save(path: Path, ep: int) -> None:
        torch.save({"learner": learner.state_dict(), "episode": ep}, path)

    total = int(tcfg["episodes"])
    for ep in range(start, total):
        obs, info = env.reset(seed=10_000 + args.seed * 1_000_000 + ep)
        state, slots, reward_sum, losses = info["state"], [], 0.0, {}
        while True:
            actions = learner.act(obs, state, explore=True)
            nobs, rew, term, trunc, info = env.step(actions)
            done = bool(term or trunc)
            learner.observe(obs, state, rew, nobs, info["state"], done)
            for k, v in learner.update().items():
                losses[k] = losses.get(k, 0.0) + float(v)
            reward_sum += float(np.mean(rew))
            slots.append(info["outcomes"])
            obs, state = nobs, info["state"]
            if done:
                break
        m = summarize_outcomes(slots)
        train_rows.append({"episode": ep, "reward_sum": reward_sum, "success_rate": m["success_rate"],
                           "mean_latency_ms": m["mean_latency_ms"], "mean_energy_mw": m["mean_energy_mw"],
                           "losses": json.dumps({k: v / len(slots) for k, v in losses.items()})})
        n = ep + 1
        if n % int(tcfg["eval_every"]) == 0:
            eval_rows.append({"episode": n, **evaluate(env, learner, int(tcfg["eval_episodes"]))})
        if n % int(tcfg["ckpt_every"]) == 0 or n == total:
            save(ckpt_path, n)
            write_rows(out / "train_log.csv", TRAIN_COLS, train_rows)
            write_rows(out / "eval_log.csv", EVAL_COLS, eval_rows)
    write_rows(out / "train_log.csv", TRAIN_COLS, train_rows)
    write_rows(out / "eval_log.csv", EVAL_COLS, eval_rows)
    save(out / "final.pt", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
