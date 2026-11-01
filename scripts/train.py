"""Train a SAC / REDQ-SAC agent on the UAV resource-allocation env.

Usage:
    python scripts/train.py --config configs/default.yaml --agent sac \
        --seed 0 --steps 300000 --ckpt-dir results/ckpt/sac_seed0

Phase 3 (Pure SAC) and Phase 4 (REDQ-SAC ensemble) both use this script.
N_critics comes from configs/default.yaml (sac.N_critics) unless
``--n-critics`` is given.  N=2 gives standard SAC (twin critics); N=10 gives
REDQ-SAC.  Use ``--n-critics`` so a single config drives both phases.

Features
--------
* TensorBoard logging (reward, critic/actor loss, alpha, eval metrics)
* Checkpoint every `training.ckpt_every` steps with `--resume` support
    (for Colab sessions that disconnect)
* Domain randomization during training (arrival rates ±20%, shadowing up to
    +3 dB, emergency surge up to 1.5x) -- the nominal training distribution
* Fixed-seed evaluation on nominal, surge, shadow, compound and unseen
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.agents.replay import ReplayBuffer
from uavrl.agents.sac import SACAgent
from uavrl.env.scenarios import Scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.eval.runner import run_episode

# --------------------------------------------------------------------------- scenario helpers
EVAL_SCENARIOS = {
    "nominal": {
        "disturbance": {"type": "none"},
        "ood": False,
    },
    "surge": {
        "disturbance": {
            "type": "surge",
            "surge_multiplier": 4.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "shadow": {
        "disturbance": {
            "type": "shadow",
            "extra_db": 15.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "compound": {
        "disturbance": {
            "type": "compound",
            "surge_multiplier": 4.0,
            "shadow_db": 15.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
    "unseen": {
        "disturbance": {
            "type": "compound",
            "surge_multiplier": 8.0,
            "shadow_db": 25.0,
            "duration_slots": 400,
            "start_slot": [500, 1200],
        },
        "ood": True,
    },
}


def make_scenario(name, rng):
    spec = EVAL_SCENARIOS[name]
    return Scenario(name, spec["disturbance"], ood=spec["ood"], rng=rng)


# --------------------------------------------------------------------------- evaluation
def evaluate(agent, env_cfg, scenario_name, n_episodes=20, seed0=10000):
    """Run n_episodes on a scenario with DETERMINISTIC actions; return metrics."""
    scenario = make_scenario(scenario_name, np.random.default_rng(seed0))
    env = UAVEnv(config=env_cfg, scenario=scenario, seed=seed0)
    summaries = []
    for i in range(n_episodes):
        _, summary = run_episode(
            env, lambda o, i: agent.select_action(o, deterministic=True), seed=seed0 + i
        )
        summaries.append(summary)

    keys = [
        "sos_violation_rate", "sos_p99_delay_s", "mean_reward",
        "mean_rate_emg_bps", "mean_rate_imp_bps", "mean_rate_norm_bps",
        "total_drops_emg_bits",
    ]
    out = {}
    for k in keys:
        vals = [s[k] for s in summaries]
        out[f"{scenario_name}/mean_{k}"] = float(np.mean(vals))
        out[f"{scenario_name}/std_{k}"] = float(np.std(vals))
    return out


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Train SAC / REDQ-SAC on the UAV env")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent", default="sac", choices=["sac"])
    parser.add_argument("--n-critics", type=int, default=None,
                        help="override sac.N_critics (2 = Pure SAC, 10 = REDQ-SAC)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--ckpt-dir", default="results/ckpt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-every", type=int, default=None,
                        help="override training.ckpt_every (use 5000 on Colab)")
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--tb-dir", default="results/tb")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- envs
    train_env = UAVEnv(config=cfg, seed=args.seed)
    eval_env_cfg = cfg  # evaluation uses the nominal config (no domain random)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    sac_cfg = cfg.get("sac", {})
    _h = sac_cfg.get("hidden", 256)
    hidden = int(_h[0] if isinstance(_h, list) else _h)
    n_critics = int(args.n_critics if args.n_critics is not None
                    else sac_cfg.get("N_critics", 10))
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        N_critics=n_critics,
        gamma=float(sac_cfg.get("gamma", 0.99)),
        lr=float(sac_cfg.get("lr", 3e-4)),
        tau=float(sac_cfg.get("tau", 0.005)),
        hidden=hidden,
        M=2,
        G=int(sac_cfg.get("G", 1)),
        rng=rng,
    )

    run_name = f"{args.agent}_seed{args.seed}_N{n_critics}"
    start_step = 1
    if args.resume:
        ckpt = os.path.join(args.ckpt_dir, f"{run_name}_latest.pt")
        if os.path.exists(ckpt):
            payload = agent.load(ckpt)
            saved_step = payload.get("step")
            if saved_step is not None:
                start_step = int(saved_step) + 1
            else:
                # Old checkpoint format: only model weights were saved.
                print("[resume] checkpoint has no step counter; restarting from step 1")
            print(f"[resume] loaded {ckpt} (next step {start_step})")
        else:
            print(f"[resume] checkpoint {ckpt} not found; starting from scratch")
    if start_step > args.steps:
        print(f"Nothing to do: checkpoint already at step {start_step - 1} "
              f">= requested {args.steps}")
        return

    # ---- logging
    tb = SummaryWriter(os.path.join(args.tb_dir, run_name))
    os.makedirs(args.ckpt_dir, exist_ok=True)

    dr_cfg = cfg.get("training", {}).get("domain_randomization", {})
    ckpt_every = int(
        args.ckpt_every
        if args.ckpt_every is not None
        else cfg.get("training", {}).get("ckpt_every", 10000)
    )
    batch = int(sac_cfg.get("batch", 256))

    replay_buffer = ReplayBuffer(
        obs_dim, act_dim, size=int(sac_cfg.get("buffer_size", 1e6)), rng=rng
    )

    # ---- training loop
    obs, _ = train_env.reset(seed=args.seed + start_step)
    train_env.randomize_episode(rng, dr_cfg)
    ep_ret = 0.0
    ep_len = 0
    t0 = time.time()
    if start_step > 1:
        print(f"[resume] replay buffer is rebuilt from step {start_step}; "
              "optimizer/model state was restored from the checkpoint")

    for step in range(start_step, args.steps + 1):
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        replay_buffer.store(
            obs, action, reward, next_obs, float(terminated or truncated)
        )

        obs = next_obs
        ep_ret += reward
        ep_len += 1

        # learn once buffer has enough samples
        if len(replay_buffer) >= batch:
            logs = agent.update(replay_buffer, batch_size=batch)
            tb.add_scalar("train/critic_loss", logs["critic_loss"], step)
            tb.add_scalar("train/actor_loss", logs["actor_loss"], step)
            tb.add_scalar("train/alpha", logs["alpha"], step)

        if terminated or truncated:
            tb.add_scalar("train/ep_return", ep_ret, step)
            tb.add_scalar("train/ep_len", ep_len, step)
            obs, _ = train_env.reset(seed=args.seed + step)
            train_env.randomize_episode(rng, dr_cfg)
            ep_ret = 0.0
            ep_len = 0

        # evaluation
        if step % args.eval_every == 0 or step == args.steps:
            for sname in EVAL_SCENARIOS:
                metrics = evaluate(
                    agent, eval_env_cfg, sname,
                    n_episodes=args.eval_episodes, seed0=10000 + step,
                )
                for k, v in metrics.items():
                    tb.add_scalar(f"eval/{k}", v, step)
            sos = {}
            for s in ["nominal", "surge", "shadow"]:
                m = evaluate(
                    agent, eval_env_cfg, s,
                    n_episodes=5, seed0=99999,
                )
                sos[s] = m[f"{s}/mean_sos_violation_rate"]
            print(
                f"step {step:6d}  "
                + "  ".join(f"{k}={v*100:5.1f}%" for k, v in sos.items())
                + f"  ({time.time()-t0:4.0f}s)"
            )
            t0 = time.time()

        # checkpoint
        if step % ckpt_every == 0 or step == args.steps:
            ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}_step{step}.pt")
            agent.save(ckpt_path, step=step)
            latest = os.path.join(args.ckpt_dir, f"{run_name}_latest.pt")
            agent.save(latest, step=step)

    tb.close()
    print(f"Training done: {run_name}")


if __name__ == "__main__":
    main()
