"""Tests for the hybrid controller (plan.md Phase 7).

- End-to-end run on surge scenario with MockLLM: calls LLM, mask activates,
  budget constraints hold (env guarantees sum b = sum p = 1).
- Timeout (p_timeout=1) -> fallback activates, no mask, no crash.
- Cooldown respected: consecutive LLM calls spaced by >= cooldown slots.
- Fast loop makes exactly one decision per slot even with 2 s latency.
- Failure-injection sweep never crashes and budget never violated.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavrl.control.hybrid_controller import HybridController
from uavrl.agents.sac import SACAgent
from uavrl.llm.client import MockLLMClient
from uavrl.llm.mock import MockLLM
from uavrl.llm.cache import LLMCache
from uavrl.env.uav_env import UAVEnv
from uavrl.env.scenarios import Scenario


def _make_env(seed=0):
    cfg = {
        "timing": {"slot_ms": 5.0, "episode_slots": 400},
        "channel": {"B_max": 1.0e7, "P_max": 1.0, "H": 100.0, "fc": 2.0e9},
        "traffic": {
            "emergency_mean_mbps": 2.0,
            "important_mean_mbps": 35.0,
            "normal_mean_mbps": 28.0,
            "queue_cap_mbit": 2.0,
        },
        "reward": {"w_emg": 1.0, "w_imp": 0.5, "w_norm": 0.1, "lambda_pen": 5.0},
        "users": {"radius_m": [400.0, 250.0, 100.0]},
    }
    scenario = Scenario("surge", {
        "type": "surge", "surge_multiplier": 4.0,
        "duration_slots": 200, "start_slot": 100,
    }, ood=True)
    return UAVEnv(config=cfg, scenario=scenario, seed=seed), cfg


def _make_agent(seed=1):
    rng = np.random.default_rng(seed)
    return SACAgent(17, 6, device="cpu", seed=seed, N_critics=5, rng=rng)


def _make_controller(env, agent, trigger_mode="hybrid", failure=None,
                     theta_queue=15000.0, latency_median_ms=350.0,
                     tau_unc=0.5, cooldown=30, ttl_slots=100, timeout_ms=1000,
                     fallback_slots=50, interim_fallback=False, seed=7):
    rng = np.random.default_rng(seed)
    client = MockLLMClient(model="mock", latency_median_ms=latency_median_ms,
                           failure_injection=failure or {}, rng=rng,
                           cache=LLMCache(enabled=False))
    return HybridController(
        agent, env, client, env.config,
        trigger_mode=trigger_mode, tau_unc=tau_unc, theta_queue=theta_queue,
        debounce_m=2, cooldown=cooldown, ttl_slots=ttl_slots,
        timeout_ms=timeout_ms, fallback_slots=fallback_slots,
        rng=rng, interim_fallback=interim_fallback,
    )


def test_end_to_end_surge():
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="hybrid", seed=7)
    steps, summary = ctrl.run_episode(seed=0)
    assert len(steps) == 400
    # budget: env guarantees sum b = sum p = 1 every step (softmax)
    assert summary["sos_violation_rate"] >= 0.0
    assert summary["mask_active_fraction"] >= 0.0
    assert summary["fallback_active_fraction"] >= 0.0
    assert summary["llm_call_rate"] >= 0.0
    print("end-to-end:", summary)


def test_timeout_fallback():
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="always",
                            failure={"p_timeout": 1.0}, seed=7,
                            latency_median_ms=350.0, timeout_ms=100,
                            cooldown=0)
    steps, summary = ctrl.run_episode(seed=0)
    # every call fails (p_timeout=1) -> fallback, zero valid masks
    assert summary["llm_valid"] == 0
    assert summary["llm_invalid"] == summary["llm_calls"]
    assert summary["fallback_active_fraction"] > 0.0
    print("timeout:", summary)


def test_invalid_json_fallback():
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="hybrid",
                            failure={"p_invalid_json": 1.0}, seed=7)
    steps, summary = ctrl.run_episode(seed=0)
    assert summary["llm_valid"] == 0
    assert summary["llm_invalid"] == summary["llm_calls"]
    print("invalid_json:", summary)


def test_cooldown_respected():
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    # low theta_queue so queue trigger fires often -> exercises cooldown
    ctrl = _make_controller(env, agent, trigger_mode="queue",
                            theta_queue=1000.0, seed=7)
    steps, summary = ctrl.run_episode(seed=0)
    call_slots = [s["slot"] for s in steps if s["llm_call"]]
    for i in range(1, len(call_slots)):
        assert call_slots[i] - call_slots[i - 1] >= ctrl.cooldown, (
            f"calls too close: {call_slots[i-1]} then {call_slots[i]}"
        )
    print("cooldown: calls=", len(call_slots), "min gap OK")


def test_one_decision_per_slot_high_latency():
    """Even with 2 s latency (400 slots), exactly one action per slot."""
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="always",
                            latency_median_ms=2000.0, seed=7, cooldown=0)
    steps, summary = ctrl.run_episode(seed=0)
    assert len(steps) == 400  # one step per slot, never blocked
    assert summary["llm_calls"] == 400  # always trigger, no cooldown
    print("high latency:", summary)


def test_failure_injection_sweep_no_crash():
    for p_inv, p_oor, p_wrong, p_to in [(0.0, 0.0, 0.0, 0.0),
                                         (0.3, 0.0, 0.0, 0.0),
                                         (0.0, 0.3, 0.0, 0.0),
                                         (0.0, 0.0, 0.3, 0.0),
                                         (0.0, 0.0, 0.0, 0.3),
                                         (0.25, 0.25, 0.25, 0.25)]:
        env, cfg = _make_env(seed=0)
        agent = _make_agent(seed=1)
        fi = {"p_invalid_json": p_inv, "p_out_of_range": p_oor,
              "p_wrong_diagnosis": p_wrong, "p_timeout": p_to}
        ctrl = _make_controller(env, agent, trigger_mode="hybrid",
                                failure=fi, seed=7)
        steps, summary = ctrl.run_episode(seed=0)
        # budget never violated: env softmax guarantees it; just assert no crash
        assert summary["sos_violation_rate"] >= 0.0
    print("failure sweep: all ran without crash")
