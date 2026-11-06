"""GPU smoke test: is training really on CUDA, and do the networks and batches live there?

Runs a short training loop (default 600 environment steps, so the first gradient
updates happen) with the REDQ agent on the chosen device, then checks that

  * CUDA is available (unless --device cpu),
  * every parameter of the actor, critic, target critic and entropy temperature is
    on that device,
  * sampled batches and observations are placed on that device,
  * the update losses are finite,

and reports steps per second.  Exit code 1 on any failure, so a notebook cell can
refuse to launch the full training.

    python scripts/gpu_smoke_test.py --out /content/drive/MyDrive/UAV_Research/results/gpu_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from uavrl.agents.replay import ReplayBuffer  # noqa: E402
from uavrl.agents.sac import SACAgent  # noqa: E402
from uavrl.env.uav_env import UAVEnv  # noqa: E402


def devices_of(module):
    return {p.device.type for p in module.parameters()}


def run(argv=None):
    ap = argparse.ArgumentParser(description="GPU smoke test for the SAC/REDQ training loop")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n-critics", type=int, default=10)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args(argv)

    cuda = torch.cuda.is_available()
    device = ("cuda" if cuda else "cpu") if args.device == "auto" else args.device
    result = {
        "torch": torch.__version__, "cuda_available": cuda, "requested": args.device,
        "device": device, "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "ok": False, "problems": [],
    }
    print("torch", torch.__version__, "| cuda available:", cuda,
          "| gpu:", result["gpu"], "| using:", device)
    try:
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip() or "nvidia-smi: no output")
    except Exception as exc:  # no NVIDIA tools on CPU machines
        print("nvidia-smi unavailable:", type(exc).__name__)

    if device == "cuda" and not cuda:
        result["problems"].append("CUDA requested but torch.cuda.is_available() is False")
        return _finish(result, args)

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = UAVEnv(config=cfg, seed=0)
    rng = np.random.default_rng(0)
    agent = SACAgent(obs_dim=17, act_dim=6, device=device, seed=0,
                     N_critics=args.n_critics, rng=rng)
    buf = ReplayBuffer(17, 6, size=10_000, rng=rng)

    # placement of the networks
    placement = {"actor": devices_of(agent.actor), "critic": devices_of(agent.critic),
                 "critic_target": devices_of(agent.critic_target),
                 "log_alpha": {agent.log_alpha.device.type}}
    result["placement"] = {k: sorted(v) for k, v in placement.items()}
    for name, devs in placement.items():
        if devs != {device}:
            result["problems"].append(f"{name} is on {sorted(devs)}, expected {device}")

    obs, _ = env.reset(seed=1)
    losses, n_updates = [], 0
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(args.steps):
        action = agent.select_action(obs, deterministic=False)
        nxt, reward, terminated, truncated, _ = env.step(action)
        buf.store(obs, action, reward, nxt, float(terminated))
        obs = env.reset(seed=2 + step)[0] if (terminated or truncated) else nxt
        if len(buf) >= 256:
            logs = agent.update(buf, batch_size=256)
            losses.append(float(logs["critic_loss"]))
            n_updates += 1
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    result.update(steps=args.steps, updates=n_updates, seconds=elapsed,
                  steps_per_second=args.steps / elapsed if elapsed > 0 else None)

    # placement of data that flows through the update
    probe = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
    batch_dev = torch.as_tensor(buf.sample_batch(8)["obs"], device=agent.device).device.type
    out_dev = agent.actor(probe.unsqueeze(0), deterministic=True, with_logprob=False)[0].device.type
    result["tensor_devices"] = {"observation": probe.device.type, "batch": batch_dev,
                                "actor_output": out_dev}
    for name, dev in result["tensor_devices"].items():
        if dev != device:
            result["problems"].append(f"{name} tensor is on {dev}, expected {device}")
    if n_updates == 0:
        result["problems"].append("no gradient update ran; increase --steps")
    elif not all(math.isfinite(x) for x in losses):
        result["problems"].append("a critic loss was NaN or infinite")
    if device == "cuda":
        result["cuda_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return _finish(result, args)


def _finish(result, args):
    result["ok"] = not result["problems"]
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    print("GPU SMOKE TEST:", "PASS" if result["ok"] else "FAIL -> " + "; ".join(result["problems"]))
    return result


if __name__ == "__main__":
    sys.exit(0 if run()["ok"] else 1)
