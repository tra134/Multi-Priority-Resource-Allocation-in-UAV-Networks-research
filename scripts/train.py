"""Train MADDPG or MAPPO on LinMECEnv (Lin et al. 2025 reproduction, docs/lin2025_spec.md Section 10).

A-23 hyperparameters must be approved by the HUMAN before a real training run.
Writes to --out: train_log.csv, eval_log.csv, manifest.json (current segment), manifest_history.jsonl
(append-only record of every start/resume, checkpoint and end; P-013), ckpt.pt, final.pt.
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
import hashlib
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


HISTORY = "manifest_history.jsonl"


def atomic_write(path: Path, write, binary: bool = False) -> None:
    """Call write(f) on a temp file in the same folder, fsync, then rename over `path`.
    A crash or error at any point leaves the old file intact (P-013, P-015)."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb" if binary else "w", **({} if binary else {"newline": ""})) as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, obj, default=None) -> None:
    text = json.dumps(obj, indent=2, default=default)  # fails before any file is touched
    atomic_write(path, lambda f: f.write(text))


def append_history(path: Path, rec: dict) -> None:
    """Append one JSON line (never rewrite). If a crash left a partial last line, start on a new line."""
    line = json.dumps(rec, default=str) + "\n"
    with path.open("ab+") as f:
        if f.seek(0, os.SEEK_END) > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                line = "\n" + line
        f.write(line.encode())
        f.flush()
        os.fsync(f.fileno())


def read_history(path: Path) -> tuple[list[dict], int]:
    """Return (records, number of unreadable lines, e.g. a line cut by a crash)."""
    if not path.exists():
        return [], 0
    recs, bad = [], 0
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    return recs, bad


def torch_load_episode(path: Path) -> int:
    import torch
    return int(torch.load(path, weights_only=False)["episode"])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
    def write(f):
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    atomic_write(path, write)


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
    ckpt_path, start, source = out / "ckpt.pt", 0, None
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        learner.load_state_dict(ck["learner"])
        start = ck["episode"]
        source = {"path": str(ckpt_path), "episode": start, "sha256": file_sha256(ckpt_path)}
    train_rows = read_rows(out / "train_log.csv", start) if start else []
    eval_rows = read_rows(out / "eval_log.csv", start + 1) if start else []
    total = int(tcfg["episodes"])
    # P-013: manifest.json = the current segment (atomic write); manifest_history.jsonl keeps every
    # start/resume, checkpoint and end of this folder (append only, never rewritten).
    hist_path = out / HISTORY
    old, _ = read_history(hist_path)
    segment = 1 + max((r.get("segment", -1) for r in old), default=-1)
    stamp, git = datetime.now(timezone.utc).isoformat(), git_info()
    configs = {"train_config": tcfg, "env_config": dataclasses.asdict(cfg)}
    manifest = {
        **git, "timestamp": stamp, "cli_args": vars(args), **configs, "training_seed_index": args.seed,
        "reset_seed_formula": "10000 + seed * 1000000 + episode", "eval_seeds_start": VAL_START,
        "resumed_from_episode": start, "segment": segment, "history_file": HISTORY, "llm_client": "none",
        "torch": torch.__version__, "numpy": np.__version__,
    }
    append_history(hist_path, {
        "event": "resume" if source else "start", "segment": segment, "timestamp": stamp, **git,
        "start_episode": start, "planned_end_episode": total, "source_checkpoint": source,
        # hash of all settings except the episode count (that is planned_end_episode), to spot a changed config
        "config_sha256": hashlib.sha256(json.dumps(
            {**configs, "train_config": {k: v for k, v in tcfg.items() if k != "episodes"}},
            sort_keys=True, default=str).encode()).hexdigest(),
        "cli_args": vars(args), "torch": torch.__version__, "numpy": np.__version__,
    })
    write_json_atomic(out / "manifest.json", manifest, default=str)

    def save(path: Path, ep: int) -> None:
        state = {"learner": learner.state_dict(), "episode": ep}
        atomic_write(path, lambda f: torch.save(state, f), binary=True)

    def log_event(event: str, **kw) -> None:
        append_history(hist_path, {"event": event, "segment": segment,
                                   "timestamp": datetime.now(timezone.utc).isoformat(), **kw})

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
            # P-015: logs first, then the checkpoint, so logs are never behind ckpt.pt after a crash
            # (resume drops rows at or after the checkpoint episode, so logs ahead of it are harmless)
            write_rows(out / "train_log.csv", TRAIN_COLS, train_rows)
            write_rows(out / "eval_log.csv", EVAL_COLS, eval_rows)
            save(ckpt_path, n)
            log_event("checkpoint", episode=n)
    write_rows(out / "train_log.csv", TRAIN_COLS, train_rows)
    write_rows(out / "eval_log.csv", EVAL_COLS, eval_rows)
    save(out / "final.pt", total)
    log_event("end", end_episode=total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
