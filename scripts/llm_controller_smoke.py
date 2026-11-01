"""End-to-end controller smoke test with a real (or mock) LLM (Phase 8).

Runs one short surge episode through the full pipeline: SAC actor -> hybrid
controller -> LLM arbiter -> schema validation -> safety mask -> environment.
This complements ``llm_smoke_test.py`` (which tests prompts in isolation).

Usage:
    python scripts/llm_controller_smoke.py                 # real Groq model
    python scripts/llm_controller_smoke.py --mock          # no network
    python scripts/llm_controller_smoke.py --slots 600 --scenario shadow
    python scripts/llm_controller_smoke.py --ckpt results/ckpt/..._latest.pt

Outputs:
    results/phase8/controller_smoke.json

Exit code is non-zero only if the run crashes, produces no LLM call, or (real
mode) no call produces a valid mask.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ),
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uavrl.agents.sac import SACAgent
from uavrl.config import find_project_root, llm_settings, load_env
from uavrl.control.hybrid_controller import HybridController
from uavrl.env.scenarios import Scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.llm.cache import LLMCache
from uavrl.llm.client import MockLLMClient, RealLLMClient


def make_scenario(name, seed):
    rng = np.random.default_rng(seed)
    if name == "surge":
        return Scenario("surge", {
            "type": "surge", "surge_multiplier": 4.0,
            "duration_slots": 150, "start_slot": 50,
        }, ood=True, rng=rng)
    if name == "shadow":
        return Scenario("shadow", {
            "type": "shadow", "extra_db": 15.0,
            "duration_slots": 150, "start_slot": 50,
        }, ood=True, rng=rng)
    return Scenario("nominal", {"type": "none"}, ood=False, rng=rng)


def main():
    parser = argparse.ArgumentParser(
        description="Run one short episode with the real LLM arbiter + safety layer"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--llm-config", default="configs/llm.yaml")
    parser.add_argument("--ckpt", default=None, help="optional SAC checkpoint")
    parser.add_argument(
        "--n-critics", type=int, default=None,
        help="override sac.N_critics; must match the checkpoint",
    )
    parser.add_argument(
        "--trigger-mode", default="hybrid",
        choices=["hybrid", "uncertainty", "queue", "periodic",
                 "reactive", "always", "random"],
        help="hybrid (default) = uncertainty OR queue threshold",
    )
    parser.add_argument("--tau-unc", type=float, default=None,
                        help="default: trigger.tau_unc in the config")
    parser.add_argument("--scenario", default="surge", choices=["surge", "shadow", "nominal"])
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta-queue", type=float, default=None,
                        help="default: trigger.theta_queue, else 50000 bits")
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="results/phase8")
    args = parser.parse_args()

    load_env()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("timing", {})["episode_slots"] = args.slots

    llm_cfg = llm_settings(args.llm_config)
    cache_cfg = llm_cfg.get("cache", {})
    cache = None
    if not args.no_cache and cache_cfg.get("enabled", True):
        cache = LLMCache(enabled=True, dir=cache_cfg.get("dir", "results/llm_cache"))

    if args.mock:
        client = MockLLMClient(latency_median_ms=350.0,
                               rng=np.random.default_rng(args.seed), cache=cache)
        model = "mock"
    else:
        model = args.model or llm_cfg.get("model")
        client = RealLLMClient(
            model=model,
            base_url=llm_cfg.get("base_url"),
            api_key=llm_cfg.get("api_key"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 512),
            retry_max=llm_cfg.get("retry_max", 1),
            timeout_s=llm_cfg.get("timeout_s", 30.0),
            json_mode=llm_cfg.get("response_format_json", True),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            rate_limit_retries=llm_cfg.get("rate_limit_retries", 3),
            rate_limit_max_sleep_s=llm_cfg.get("rate_limit_max_sleep_s", 30.0),
            cache=cache,
        )

    sac_cfg = cfg.get("sac", {})
    n_critics = int(
        args.n_critics
        if args.n_critics is not None
        else sac_cfg.get("N_critics", 10)
    )
    agent = SACAgent(
        obs_dim=17, act_dim=6, device="cpu", seed=args.seed,
        N_critics=n_critics,
        hidden=int((sac_cfg.get("hidden") or [256])[0]),
        rng=np.random.default_rng(args.seed),
    )
    if args.ckpt and os.path.exists(args.ckpt):
        agent.load(args.ckpt)
        print(f"loaded checkpoint {args.ckpt}")
    elif args.ckpt:
        raise FileNotFoundError(args.ckpt)

    trigger_cfg = cfg.get("trigger", {})
    tau_unc = (args.tau_unc if args.tau_unc is not None
               else trigger_cfg.get("tau_unc"))
    theta_queue = (args.theta_queue if args.theta_queue is not None
                   else trigger_cfg.get("theta_queue"))
    if theta_queue is None:
        theta_queue = 50000.0
    cooldown = (args.cooldown if args.cooldown is not None
                else int(trigger_cfg.get("cooldown", 80)))

    env = UAVEnv(config=cfg, scenario=make_scenario(args.scenario, args.seed),
                 seed=args.seed)
    ctrl = HybridController(
        agent, env, client, cfg,
        trigger_mode=args.trigger_mode, tau_unc=tau_unc,
        theta_queue=theta_queue,
        debounce_m=int(trigger_cfg.get("debounce_m", 3)),
        cooldown=cooldown,
        ttl_slots=int(trigger_cfg.get("ttl_slots", 600)),
        timeout_ms=args.timeout_ms,
        fallback_slots=int(cfg.get("fallback", {}).get("fallback_slots", 200)),
        rng=np.random.default_rng(args.seed),
    )

    t0 = time.perf_counter()
    steps, summary = ctrl.run_episode(seed=args.seed)
    wall_s = time.perf_counter() - t0

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "provider": llm_cfg.get("provider"),
        "scenario": args.scenario,
        "slots": args.slots,
        "seed": args.seed,
        "trigger_mode": args.trigger_mode,
        "tau_unc": tau_unc,
        "theta_queue": theta_queue,
        "n_critics": n_critics,
        "wall_seconds": wall_s,
        "summary": summary,
        "client_stats": getattr(client, "stats", lambda: {})(),
        "steps": steps,
    }
    out_dir = os.path.join(str(find_project_root()), args.out)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "controller_smoke.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== controller + LLM smoke test ===")
    print(f"model              : {model}")
    print(f"scenario / slots   : {args.scenario} / {args.slots}")
    print(f"trigger            : {args.trigger_mode} (tau_unc={tau_unc}, "
          f"theta_queue={theta_queue})")
    print(f"LLM calls          : {summary['llm_calls']}")
    print(f"valid / invalid    : {summary['llm_valid']} / {summary['llm_invalid']}")
    print(f"timeouts           : {summary['llm_timeouts']}")
    print(f"mask active        : {summary['mask_active_fraction']:.1%} of slots")
    print(f"fallback active    : {summary['fallback_active_fraction']:.1%} of slots")
    print(f"SOS violation rate : {summary['sos_violation_rate']:.1%}")
    print(f"wall time          : {wall_s:.1f} s")
    print(f"wrote {path}")

    if summary["llm_calls"] == 0:
        return 1
    if not args.mock and summary["llm_valid"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
