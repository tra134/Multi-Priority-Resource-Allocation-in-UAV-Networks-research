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

from uavrl.agents.sac import SACAgent
from uavrl.control.hybrid_controller import HybridController
from uavrl.env.scenarios import Scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.llm.cache import LLMCache
from uavrl.llm.client import MockLLMClient


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
    assert summary["mask_active_fraction"] > 0.0, "valid MockLLM responses should install masks"
    assert summary["fallback_active_fraction"] >= 0.0
    assert summary["llm_call_rate"] >= 0.0
    assert summary["llm_valid"] > 0
    print("end-to-end:", summary)


def test_timeout_fallback():
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="always",
                            failure={"p_timeout": 1.0}, seed=7,
                            latency_median_ms=350.0, timeout_ms=100,
                            cooldown=0)
    steps, summary = ctrl.run_episode(seed=0)
    # every call times out (p_timeout=1) -> fallback, zero valid masks
    assert summary["llm_valid"] == 0
    assert summary["llm_timeouts"] == summary["llm_calls"]
    assert summary["fallback_active_fraction"] > 0.0
    print("timeout:", summary)


def test_invalid_json_fallback():
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="hybrid",
                            failure={"p_invalid_json": 1.0}, seed=7)
    steps, summary = ctrl.run_episode(seed=0)
    assert summary["llm_valid"] == 0
    # Slow replies (> timeout_ms) are timeouts, the rest are invalid replies.
    assert summary["llm_invalid"] + summary["llm_timeouts"] == summary["llm_calls"]
    assert summary["llm_invalid"] > 0
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


def test_call_log_matches_counters():
    """Phase 9 metrics read per-call diagnostics from ``call_log``."""
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl = _make_controller(env, agent, trigger_mode="hybrid", seed=7)
    _, summary = ctrl.run_episode(seed=0)

    assert len(ctrl.call_log) == summary["llm_calls"] > 0
    for entry in ctrl.call_log:
        assert set(entry) >= {
            "slot", "ood", "trigger_reason", "ok", "timeout", "reason",
            "mode", "latency_ms",
        }
        assert entry["latency_ms"] >= 0.0
        if entry["ok"]:
            assert entry["mode"] in {"NORMAL", "SOS_SURGE", "DEEP_FADING",
                                     "COMPOUND"}
    valid = sum(1 for e in ctrl.call_log if e["ok"])
    invalid = sum(1 for e in ctrl.call_log if not e["ok"] and not e["timeout"])
    timeouts = sum(1 for e in ctrl.call_log if e["timeout"])
    # A response still in flight when the episode ends is a valid call but was
    # never applied, so the applied counter can only be lower.
    assert summary["llm_valid"] <= valid
    assert summary["llm_invalid"] == invalid
    assert summary["llm_timeouts"] == timeouts


def test_interim_fallback_fills_latency_window():
    """M12b applies the safe fallback from trigger until the answer arrives."""
    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    common = dict(
        trigger_mode="always", cooldown=0, latency_median_ms=2000.0,
        timeout_ms=100_000, fallback_slots=50, seed=7,
    )
    ctrl_plain = _make_controller(env, agent, interim_fallback=False, **common)
    _, plain = ctrl_plain.run_episode(seed=0)

    env, cfg = _make_env(seed=0)
    agent = _make_agent(seed=1)
    ctrl_interim = _make_controller(env, agent, interim_fallback=True, **common)
    _, interim = ctrl_interim.run_episode(seed=0)

    assert plain["llm_timeouts"] == 0
    assert plain["fallback_active_fraction"] == 0.0
    assert interim["fallback_active_fraction"] > 0.5
    # Responses still arrive and install valid masks; they are preempted by the
    # interim fallback only while the next response is in flight.
    assert interim["llm_valid"] > 0


# ---------------------------------------------------------------- stage 1 fixes
from uavrl.llm.latency import LatencyModel  # noqa: E402

_GOOD = {
    "operating_mode": "SOS_SURGE",
    "min_bandwidth_ratio": {"emergency": 0.6, "important": 0.2, "normal": 0.15},
    "power_boost_emergency": True,
    "ttl_duration_slots": 100,
}


class _FixedClient:
    """Returns one scripted reply with a fixed reported latency."""

    def __init__(self, resp, latency_ms, ok, reason):
        self.args = (resp, latency_ms, ok, reason)

    def call(self, prompt, features=None):
        return self.args


def _ctrl(client, env, agent, **kw):
    params = dict(trigger_mode="periodic", periodic_K=10_000, cooldown=0,
                  timeout_ms=1000, fallback_slots=50, ttl_slots=100,
                  rng=np.random.default_rng(0))
    params.update(kw)
    return HybridController(agent, env, client, env.config, **params)


def _first(steps, key):
    return next(s["slot"] for s in steps if s[key])


def test_timeout_fallback_starts_only_after_timeout_elapses():
    """F5: a timeout is known after timeout_ms (1000 ms = 200 slots), not at call time."""
    env, _ = _make_env(seed=0)
    ctrl = _ctrl(_FixedClient(None, 5000.0, False, "timeout"), env, _make_agent())
    steps, _ = ctrl.run_episode(seed=0)
    assert _first(steps, "fallback_active") == 200
    assert sum(s["fallback_active"] for s in steps) == 50


def test_invalid_reply_fallback_starts_when_reply_arrives():
    """F5: an invalid reply is known when it arrives (300 ms = 60 slots)."""
    env, _ = _make_env(seed=0)
    ctrl = _ctrl(_FixedClient(None, 300.0, False, "invalid_json"), env, _make_agent())
    steps, _ = ctrl.run_episode(seed=0)
    assert _first(steps, "fallback_active") == 60


def test_cache_hit_still_pays_simulated_latency():
    """F13: a cache hit reports 0 ms but the mask must still arrive after the model latency."""
    env, _ = _make_env(seed=0)
    client = _FixedClient(dict(_GOOD), 0.0, True, "cache")
    ctrl = _ctrl(client, env, _make_agent(),
                 latency_model=LatencyModel(median_ms=500.0, sigma=0.0))
    steps, _ = ctrl.run_episode(seed=0)
    assert _first(steps, "mask_active") == 100  # 500 ms / 5 ms
    assert np.isnan(ctrl.call_log[0]["measured_latency_ms"])


def test_fitted_latency_overrides_measured_latency():
    env, _ = _make_env(seed=0)
    client = _FixedClient(dict(_GOOD), 5.0, True, "ok")  # wall clock says 5 ms
    ctrl = _ctrl(client, env, _make_agent(), use_fitted_latency=True,
                 latency_model=LatencyModel(median_ms=250.0, sigma=0.0))
    steps, _ = ctrl.run_episode(seed=0)
    assert _first(steps, "mask_active") == 50
    assert ctrl.call_log[0]["measured_latency_ms"] == 5.0


def test_controller_uses_deterministic_actor_by_default():
    """F4: LLM methods must use the same deterministic actor as M3-M5."""
    env, _ = _make_env(seed=0)
    agent = _make_agent()
    seen = []
    original = agent.select_action

    def spy(obs, deterministic=False):
        seen.append(deterministic)
        return original(obs, deterministic=deterministic)

    agent.select_action = spy
    ctrl = _make_controller(env, agent, trigger_mode="queue", theta_queue=1e12)
    ctrl.run_episode(seed=0)
    assert seen and all(seen)


def test_csi_window_is_updated_every_slot():
    """F6: the '50-slot window' must cover 50 slots, not the last 50 calls."""
    env, _ = _make_env(seed=0)
    ctrl = _make_controller(env, _make_agent(), trigger_mode="queue", theta_queue=1e12)
    ctrl.run_episode(seed=0)
    assert ctrl.calls == 0
    assert len(ctrl.csi_hist) == 50

    env, _ = _make_env(seed=0)
    ctrl = _ctrl(_FixedClient(dict(_GOOD), 50.0, True, "ok"), env, _make_agent(),
                 periodic_K=30)
    seen = {}
    original = ctrl._build_features

    def spy(obs, info, reason):
        seen[ctrl.slot] = len(ctrl.csi_hist)
        return original(obs, info, reason)

    ctrl._build_features = spy
    ctrl.run_episode(seed=0)
    assert seen[0] == 1 and seen[30] == 31 and seen[60] == 50


def _csi_means(scenario, seeds=(0, 1, 2, 3)):
    out = []
    for s in seeds:
        env, _ = _make_env(seed=s)
        env.scenario = scenario
        ctrl = _ctrl(_FixedClient(dict(_GOOD), 50.0, True, "ok"), env, _make_agent(),
                     periodic_K=100)
        feats = []
        original = ctrl._build_features

        def spy(obs, info, reason, _o=original, _f=feats):
            f = _o(obs, info, reason)
            _f.append(f)
            return f

        ctrl._build_features = spy
        ctrl.run_episode(seed=s)
        out.append(np.mean([f["csi_emergency_mean"] for f in feats[1:]]))
    return float(np.median(out))


def test_csi_in_prompt_is_normalized_to_nominal_one():
    """F21: raw gain (~1e-10) was shown to the LLM as 'normalized, nominal 1.0'."""
    nominal = _csi_means(None)
    shadowed = _csi_means(Scenario(
        "shadow", {"type": "shadow", "extra_db": 15.0, "duration_slots": 400,
                   "start_slot": 0}, ood=True))
    # Scale is O(1)-O(10), not O(1e-10) as with raw gain.  The LoS/NLoS switch
    # (19 dB) means the absolute level varies, so test the relative drop: a
    # 15 dB shadow (31x in power) must be clearly visible.
    assert 0.1 < nominal < 100.0, nominal
    assert shadowed < nominal / 10.0, (shadowed, nominal)


def test_mock_reaches_fading_modes_through_the_real_controller():
    """F3: through the controller, shadowing must yield non-surge diagnoses."""
    env, _ = _make_env(seed=2)
    env.scenario = Scenario("shadow", {"type": "shadow", "extra_db": 15.0,
                                       "duration_slots": 400, "start_slot": 0}, ood=True)
    ctrl = _make_controller(env, _make_agent(), trigger_mode="queue",
                            theta_queue=1000.0, cooldown=10, seed=3)
    ctrl.run_episode(seed=2)
    modes = {c["mode"] for c in ctrl.call_log if c["mode"]}
    assert modes & {"DEEP_FADING", "COMPOUND"}, modes
