"""Baselines: queue-aware priority (F1), Rule-Shield on SAC (F2), M13 reservation."""

import numpy as np

from uavrl.baselines.rule_shield import FixedReservation, RuleShield
from uavrl.baselines.static_policies import (
    QueueAwarePriority,
    emergency_only,
    shares_to_action,
)
from uavrl.env.scenarios import Scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.eval.runner import run_episode
from uavrl.llm.latency import LatencyModel

GOOD_GAIN = [1e-11, 1e-11, 1e-11]


def _shares(action):
    e = np.exp(3.0 * np.asarray(action[:3], dtype=float))
    return e / e.sum()


def test_empty_queues_do_not_starve_other_classes():
    """F1: the old policy gave Important/Normal ~0 even with nothing queued."""
    pol = QueueAwarePriority()
    info = {"queue_bits": [0.0, 0.0, 0.0], "gain": GOOD_GAIN}
    s = _shares(pol(None, info))
    assert s[1] > 0.1 and s[2] > 0.1
    assert abs(s.sum() - 1.0) < 1e-9
    old = _shares(emergency_only(None, info))
    assert old[1] < 1e-3 and old[2] < 1e-3


def test_large_emergency_queue_takes_priority():
    pol = QueueAwarePriority()
    info = {"queue_bits": [2.0e6, 5.0e5, 5.0e5], "gain": GOOD_GAIN}
    s = _shares(pol(None, info))
    assert s[0] > s[1] and s[0] > s[2]


def test_works_before_first_info():
    s = _shares(QueueAwarePriority()(None, {}))
    assert abs(s.sum() - 1.0) < 1e-9 and np.all(s > 0)


def _env(seed=0, slots=400):
    cfg = {
        "timing": {"slot_ms": 5.0, "episode_slots": slots},
        "channel": {"B_max": 1.0e7, "P_max": 1.0, "H": 100.0, "fc": 2.0e9},
        "traffic": {"emergency_mean_mbps": 2.0, "important_mean_mbps": 35.0,
                    "normal_mean_mbps": 28.0, "queue_cap_mbit": 2.0},
        "reward": {"w_emg": 1.0, "w_imp": 0.5, "w_norm": 0.1, "lambda_pen": 5.0},
        "users": {"radius_m": [400.0, 250.0, 100.0]},
    }
    sc = Scenario("surge", {"type": "surge", "surge_multiplier": 4.0,
                            "duration_slots": 200, "start_slot": 100}, ood=True)
    return UAVEnv(config=cfg, scenario=sc, seed=seed), cfg


def test_queue_aware_priority_delivers_important_traffic():
    env, cfg = _env()
    steps, summary = run_episode(env, QueueAwarePriority.from_config(cfg), seed=0)
    assert summary["mean_rate_imp_bps"] > 1.0e6  # emergency_only gives ~1e5 bps


def _base(obs, info):
    return shares_to_action([0.1, 0.45, 0.45], [0.1, 0.45, 0.45])


def test_rule_shield_wraps_any_base_policy():
    """F2: the shield must be able to sit on top of the learned policy."""
    env, _ = _env()
    shield = RuleShield(theta_queue_bits=5000.0, m=3, ttl_slots=100, cooldown=20,
                        base_policy=_base)
    steps, _ = run_episode(env, shield, seed=0)
    assert shield.mask_active_slots > 0
    # with a mask active the Emergency share is at least the floor
    masked = [s for s in steps if s["slot"] > 0]
    assert masked


def test_rule_shield_fitted_latency_delays_mask_and_ttl_starts_on_application():
    env, _ = _env(slots=600)
    shield = RuleShield(theta_queue_bits=5000.0, m=3, ttl_slots=100, cooldown=20,
                        base_policy=_base,
                        latency_model=LatencyModel(median_ms=500.0, sigma=0.0),
                        rng=np.random.default_rng(0))
    run_episode(env, shield, seed=0)
    assert shield.trigger_count >= 1
    # 500 ms = 100 slots between trigger and application; TTL 100 starts then.
    assert shield.mask_active_slots <= 100 * shield.trigger_count


def test_fixed_reservation_keeps_emergency_floor():
    pol = FixedReservation(_base, bw_floor_emg=0.5, power_floor_emg=0.5)
    a = pol(None, {})
    assert _shares(a)[0] >= 0.5 - 1e-6
    assert pol.mask.valid()


def _env_shares(v):
    """Shares the environment derives from an action: softmax(3 * x) (uav_env.py)."""
    z = 3.0 * np.asarray(v, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def test_rule_shield_without_mask_leaves_the_actor_allocation_unchanged():
    """P-018: with no active mask the shield must hand the environment the same
    bandwidth and power shares as the base policy (it used softmax(x), not softmax(3x))."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        shield = RuleShield(theta_queue_bits=1e18, base_policy=lambda obs, info, _a=a: _a)
        out = shield(None, {})
        assert np.allclose(_env_shares(out[:3]), _env_shares(a[:3]), atol=1e-6)
        assert np.allclose(_env_shares(out[3:]), _env_shares(a[3:]), atol=1e-6)


def test_fixed_reservation_with_zero_floors_leaves_the_actor_allocation_unchanged():
    """P-018, M13: a reservation whose floors are zero is the actor itself."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        pol = FixedReservation(lambda obs, info, _a=a: _a, bw_floor_emg=0.0, power_floor_emg=0.0)
        out = pol(None, {})
        assert np.allclose(_env_shares(out[:3]), _env_shares(a[:3]), atol=1e-6)
        assert np.allclose(_env_shares(out[3:]), _env_shares(a[3:]), atol=1e-6)


def _untrained_agent():
    from uavrl.agents.sac import SACAgent

    return SACAgent(17, 6, device="cpu", seed=1, N_critics=2, rng=np.random.default_rng(1))


def _short_env(seed=0):
    from tests.test_controller import _make_env

    return _make_env(seed=seed)[0]


def test_rule_shield_without_mask_reproduces_the_actor_trajectory():
    """P-019: the actor reads its previous action in the observation, so a shield that
    re-encodes the action (same shares, different vector) changes later decisions."""
    agent = _untrained_agent()

    def base(obs, info):
        return agent.select_action(obs, deterministic=True)

    plain, _ = run_episode(_short_env(), base, seed=0)
    shield = RuleShield(theta_queue_bits=1e18, base_policy=base)
    shielded, _ = run_episode(_short_env(), shield, seed=0)
    assert shield.mask_active_slots == 0
    assert np.allclose([s["reward"] for s in plain], [s["reward"] for s in shielded], atol=1e-9)
    assert np.allclose([s["b_share_emg"] for s in plain],
                       [s["b_share_emg"] for s in shielded], atol=1e-9)


def test_fixed_reservation_whose_floors_the_actor_already_meets_leaves_it_unchanged():
    """P-019, M13: a projection that changes nothing must return the actor's own vector."""
    a = np.array([0.9, -0.1, -0.2, 0.8, 0.1, -0.3], dtype=np.float32)   # not zero-mean, meets the floors
    pol = FixedReservation(lambda obs, info, _a=a: _a, bw_floor_emg=0.2, power_floor_emg=0.2)
    out = pol(None, {})
    assert np.array_equal(out, a)
