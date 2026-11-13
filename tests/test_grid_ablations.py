"""E5 failure injection and E3 trigger ablation (plan.md Stage 6)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_controller import _make_agent, _make_controller, _make_env

from uavrl.control.hybrid_controller import HybridController
from uavrl.llm.cache import LLMCache
from uavrl.llm.client import MockLLMClient

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rg", _ROOT / "scripts" / "run_grid.py")
rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rg)


# ------------------------------------------------------------------ E5
@pytest.mark.parametrize("kind", list(rg.FAILURE_KINDS))
def test_failure_prob_one_each_kind_is_handled(kind):
    inj = rg.failure_injection_for(1.0, kind)
    assert inj == {rg.FAILURE_KINDS[kind]: 1.0}
    env, _ = _make_env(seed=0)
    ctrl = _make_controller(env, _make_agent(1), trigger_mode="always",
                            failure=inj, cooldown=0, timeout_ms=1000, seed=7)
    steps, summary = ctrl.run_episode(seed=0)
    assert len(steps) == 400
    assert summary["llm_calls"] > 0
    # the env projects every action onto the budget, so b and p shares sum to 1
    for s in steps:
        assert abs(s["b_sum"] - 1.0) < 1e-6
        assert abs(s["p_sum"] - 1.0) < 1e-6
    if kind in ("invalid_json", "timeout"):
        assert summary["llm_valid"] == 0
        assert summary["llm_invalid"] + summary["llm_timeouts"] == summary["llm_calls"]
        assert summary["fallback_active_fraction"] > 0.0
    elif kind == "out_of_range":
        # out-of-range floors are rejected when they arrive: counted invalid,
        # which starts the safety fallback (valid is counted on arrival, so
        # the three counters are not a partition of llm_calls)
        assert summary["llm_invalid"] > 0
        assert summary["fallback_active_fraction"] > 0.0
    else:  # wrong_diagnosis: schema-valid reply with the wrong mode
        assert summary["llm_valid"] > 0
        assert summary["llm_invalid"] == 0


def test_failure_prob_zero_equals_default():
    assert rg.failure_injection_for(0.0, "timeout") == {}
    assert rg.failure_injection_for(0.0, None) == {}
    runs = []
    agent = _make_agent(1)  # one agent: network init is not seeded per instance
    for failure in (None, {"p_timeout": 0.0}):
        env, _ = _make_env(seed=0)
        ctrl = _make_controller(env, agent, failure=failure, seed=7)
        steps, summary = ctrl.run_episode(seed=0)
        runs.append((summary, [c["slot"] for c in ctrl.call_log]))
    assert runs[0] == runs[1]


def test_failure_flags_refused_with_real_llm():
    with pytest.raises(SystemExit):
        rg.check_failure_args(0.5, "timeout", "real")
    with pytest.raises(SystemExit):
        rg.check_failure_args(0.5, None, "mock")
    with pytest.raises(SystemExit):
        rg.check_failure_args(1.5, "timeout", "mock")
    rg.check_failure_args(0.0, None, "real")  # default stays allowed
    rg.check_failure_args(0.5, "timeout", "mock")


# ------------------------------------------------------------------ E3
def _ctrl(trigger_mode, u_fn, tau_unc, theta_queue, debounce_m):
    env, _ = _make_env(seed=0)
    agent = _make_agent(1)
    state = {"t": 0}
    real_select = agent.select_action

    def select(obs, deterministic=True):
        state["t"] += 1
        return real_select(obs, deterministic=deterministic)

    agent.select_action = select
    agent.uncertainty = lambda obs, a: (u_fn(state["t"] - 1), 0.0)
    client = MockLLMClient(latency_median_ms=350.0, rng=np.random.default_rng(7),
                           cache=LLMCache(enabled=False))
    return HybridController(
        agent, env, client, env.config, trigger_mode=trigger_mode,
        tau_unc=tau_unc, theta_queue=theta_queue, debounce_m=debounce_m,
        cooldown=0, ttl_slots=100, timeout_ms=1000, fallback_slots=50,
        rng=np.random.default_rng(7))


def test_resolve_trigger_only_touches_hybrid():
    assert rg.resolve_trigger("hybrid", "none") == "hybrid"
    assert rg.resolve_trigger("hybrid", "unc_only") == "uncertainty"
    assert rg.resolve_trigger("hybrid", "queue_only") == "queue"
    for t in ("periodic", "random", "reactive", "always"):
        assert rg.resolve_trigger(t, "unc_only") == t


def test_unc_only_ignores_queue_condition():
    # queue condition always true (theta=-1), uncertainty never exceeds tau
    ctrl = _ctrl("uncertainty", lambda t: 0.0, tau_unc=0.5, theta_queue=-1.0,
                 debounce_m=1)
    ctrl.run_episode(seed=0)
    assert ctrl.call_log == []


def test_queue_only_ignores_uncertainty_condition():
    ctrl = _ctrl("queue", lambda t: 9.0, tau_unc=0.5, theta_queue=1e18,
                 debounce_m=1)
    ctrl.run_episode(seed=0)
    assert ctrl.call_log == []


def test_hybrid_still_fires_on_either_condition():
    c1 = _ctrl("hybrid", lambda t: 9.0, tau_unc=0.5, theta_queue=1e18, debounce_m=1)
    c1.run_episode(seed=0)
    c2 = _ctrl("hybrid", lambda t: 0.0, tau_unc=0.5, theta_queue=-1.0, debounce_m=1)
    c2.run_episode(seed=0)
    assert c1.call_log and c2.call_log


def test_debounce_off_triggers_on_single_exceeding_slot():
    def spike(t):
        return 9.0 if t == 50 else 0.0

    on = _ctrl("uncertainty", spike, tau_unc=0.5, theta_queue=None, debounce_m=3)
    on.run_episode(seed=0)
    assert on.call_log == []
    off = _ctrl("uncertainty", spike, tau_unc=0.5, theta_queue=None, debounce_m=1)
    off.run_episode(seed=0)
    assert [c["slot"] for c in off.call_log] == [50]


def test_default_hybrid_decisions_are_deterministic_and_ablation_none_is_noop():
    runs = []
    agent = _make_agent(1)
    for _ in range(2):
        env, _ = _make_env(seed=0)
        ctrl = _make_controller(env, agent, trigger_mode="hybrid", seed=7)
        steps, summary = ctrl.run_episode(seed=0)
        runs.append((summary, [c["slot"] for c in ctrl.call_log]))
    assert runs[0] == runs[1]
    assert rg.resolve_trigger("hybrid", "none") == "hybrid"
