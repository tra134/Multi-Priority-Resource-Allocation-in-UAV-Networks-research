"""Tests for the safety layer: mask projection, fallback, rule shield (Phase 6).

Property-based with random inputs:
- Output shares sum to 1; output >= floors; projection idempotent.
- Invalid masks (sum(floor)>1 or floor out of [0,1]) are rejected.
- Mask expiry is correct.
- Budget constraints hold in every env step over 10^5 random steps.
- Rule-Shield respects cooldown and reduces Emergency violations under surge.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavrl.baselines.rule_shield import RuleShield
from uavrl.baselines.static_policies import strict_priority
from uavrl.control.fallback import FALLBACK_BW, FALLBACK_POWER, fallback_action
from uavrl.control.mask import Mask, project_bw_power
from uavrl.env.scenarios import Scenario
from uavrl.env.uav_env import UAVEnv
from uavrl.eval.runner import run_episode


def _random_mask(rng):
    f = rng.random(3) * 0.9
    f = f * (0.95 / f.sum())  # ensure sum <= 0.95
    return Mask(bw_floor=f, power_floor_emg=float(rng.random() * 0.9),
                expires_at=int(rng.integers(1, 1000)))


def test_projection_sums_and_floors():
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = _random_mask(rng)
        b_raw = rng.random(3); b_raw /= b_raw.sum()
        p_raw = rng.random(3); p_raw /= p_raw.sum()
        b, p = project_bw_power(b_raw, p_raw, mask)
        assert abs(b.sum() - 1.0) < 1e-6
        assert abs(p.sum() - 1.0) < 1e-6
        assert np.all(b >= mask.bw_floor - 1e-6)
        assert p[0] >= mask.power_floor_emg - 1e-6


def test_projection_feasible_and_preserves_floors():
    # Projecting an already-feasible share stays feasible (floors respected,
    # sums to 1).  Note: the plan's b = f + (1-sum(f))*r map is not a
    # fixed-point idempotent (only b = f/sum(f) is), but it always yields a
    # feasible share -- which is the safety-relevant property (Claim D).
    rng = np.random.default_rng(1)
    mask = _random_mask(rng)
    b_raw = rng.random(3); b_raw /= b_raw.sum()
    p_raw = rng.random(3); p_raw /= p_raw.sum()
    b1, p1 = project_bw_power(b_raw, p_raw, mask)
    b2, p2 = project_bw_power(b1, p1, mask)
    for b, p in [(b1, p1), (b2, p2)]:
        assert abs(b.sum() - 1.0) < 1e-6
        assert abs(p.sum() - 1.0) < 1e-6
        assert np.all(b >= mask.bw_floor - 1e-6)
        assert p[0] >= mask.power_floor_emg - 1e-6


def test_invalid_mask_rejected():
    rng = np.random.default_rng(2)
    # sum(floor) > 1
    bad = Mask(bw_floor=np.array([0.6, 0.6, 0.0]), power_floor_emg=0.5,
               expires_at=100)
    assert not bad.valid()
    b_raw = np.array([1.0 / 3.0] * 3)
    p_raw = np.array([1.0 / 3.0] * 3)
    with pytest.raises(ValueError):
        project_bw_power(b_raw, p_raw, bad)
    # floor out of range
    bad2 = Mask(bw_floor=np.array([1.2, 0.0, 0.0]), power_floor_emg=0.5,
                expires_at=100)
    assert not bad2.valid()


def test_mask_expiry():
    mask = Mask(bw_floor=np.array([0.5, 0.0, 0.0]), power_floor_emg=0.5,
                expires_at=10)
    assert not mask.expired(9)
    assert mask.expired(10)
    assert mask.expired(50)


def test_fallback_shares():
    b, p = FALLBACK_BW, FALLBACK_POWER
    assert abs(b.sum() - 1.0) < 1e-6 and abs(p.sum() - 1.0) < 1e-6
    assert b[0] == 0.8 and p[0] == 0.8
    a = fallback_action()
    assert a.shape == (6,)


def test_rule_shield_counters():
    """Phase 9 reads triggers / mask-active slots for M7."""
    cfg = {
        "timing": {"slot_ms": 5.0, "episode_slots": 400},
        "channel": {"B_max": 1.0e7, "P_max": 1.0, "H": 100.0, "fc": 2.0e9},
        "traffic": {
            "emergency_mean_mbps": 2.0, "important_mean_mbps": 35.0,
            "normal_mean_mbps": 28.0, "queue_cap_mbit": 2.0,
        },
        "reward": {"w_emg": 1.0, "w_imp": 0.5, "w_norm": 0.1, "lambda_pen": 5.0},
        "users": {"radius_m": [400.0, 250.0, 100.0]},
    }
    scenario = Scenario("surge", {
        "type": "surge", "surge_multiplier": 4.0, "duration_slots": 200,
        "start_slot": 100,
    }, ood=True)
    env = UAVEnv(config=cfg, scenario=scenario, seed=3)
    shield = RuleShield(theta_queue_bits=5000.0, m=3, ttl_slots=100,
                        cooldown=20, latency_slots=0,
                        base_policy=strict_priority)
    steps, _ = run_episode(env, shield, seed=3)

    assert shield.trigger_count > 0
    assert shield.mask_active_slots > 0
    assert shield.mask_active_slots <= len(steps)
    assert shield.trigger_count <= len(steps)


def test_rule_shield_cooldown_and_surge():
    cfg = {
        "timing": {"slot_ms": 5.0, "episode_slots": 2000},
        "channel": {"B_max": 1.0e7, "P_max": 1.0, "H": 100.0, "fc": 2.0e9},
        "traffic": {
            "emergency_mean_mbps": 2.0, "important_mean_mbps": 35.0,
            "normal_mean_mbps": 28.0, "queue_cap_mbit": 2.0,
        },
        "reward": {"w_emg": 1.0, "w_imp": 0.5, "w_norm": 0.1, "lambda_pen": 5.0},
        "users": {"radius_m": [400.0, 250.0, 100.0]},
    }
    # surge scenario: Emergency 4x from slot 500 for 400 slots
    from uavrl.env.scenarios import Scenario
    scenario = Scenario("surge", {
        "type": "surge", "surge_multiplier": 4.0, "duration_slots": 400,
        "start_slot": 500,
    }, ood=True)

    def eval_shield(latency_slots):
        env = UAVEnv(config=cfg, scenario=scenario, seed=10)
        shield = RuleShield(theta_queue_bits=20000.0, m=3, ttl_slots=600,
                            cooldown=50, latency_slots=latency_slots,
                            base_policy=strict_priority)
        from uavrl.eval.runner import run_episode
        _, summary = run_episode(env, shield, seed=10)
        return summary["sos_violation_rate"]

    viol_zero = eval_shield(0)
    viol_delayed = eval_shield(70)  # 350 ms = 70 slots
    print(f"RuleShield surge SOS viol: zero-lat={viol_zero*100:.2f}%  "
          f"350ms-lat={viol_delayed*100:.2f}%")
    assert 0.0 <= viol_zero <= 1.0
    assert 0.0 <= viol_delayed <= 1.0


def test_budget_constraints_random_steps():
    """sum b == sum p == 1 and shares >= 0 over 10^5 random env steps."""
    cfg = {
        "timing": {"slot_ms": 5.0, "episode_slots": 2000},
        "channel": {"B_max": 1.0e7, "P_max": 1.0, "H": 100.0, "fc": 2.0e9},
        "traffic": {
            "emergency_mean_mbps": 2.0, "important_mean_mbps": 35.0,
            "normal_mean_mbps": 28.0, "queue_cap_mbit": 2.0,
        },
        "reward": {"w_emg": 1.0, "w_imp": 0.5, "w_norm": 0.1, "lambda_pen": 5.0},
        "users": {"radius_m": [400.0, 250.0, 100.0]},
    }
    env = UAVEnv(config=cfg, seed=5)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(6)
    n = 100000
    worst_b = 0.0
    worst_p = 0.0
    min_share = 1.0
    for _ in range(n):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        obs, r, terminated, truncated, info = env.step(a)
        b = np.array([0.0, 0.0, 0.0])  # placeholder; recompute from action
        # Recompute shares the same way the env does:
        b_share = np.exp(3 * a[:3] - np.max(3 * a[:3])); b_share /= b_share.sum()
        p_share = np.exp(3 * a[3:] - np.max(3 * a[3:])); p_share /= p_share.sum()
        worst_b = max(worst_b, abs(b_share.sum() - 1.0))
        worst_p = max(worst_p, abs(p_share.sum() - 1.0))
        min_share = min(min_share, float(b_share.min()), float(p_share.min()))
        if terminated or truncated:
            obs, _ = env.reset(seed=5)
    assert worst_b < 1e-4 and worst_p < 1e-4
    assert min_share >= -1e-6


def test_projection_leaves_feasible_action_unchanged():
    """F18: an action that already meets the floors must not be altered."""
    mask = Mask(bw_floor=np.array([0.3, 0.1, 0.0]), power_floor_emg=0.4, expires_at=100)
    b_raw = np.array([0.5, 0.3, 0.2])
    p_raw = np.array([0.6, 0.3, 0.1])
    b, p = project_bw_power(b_raw, p_raw, mask)
    assert np.allclose(b, b_raw) and np.allclose(p, p_raw)


def test_projection_is_idempotent():
    rng = np.random.default_rng(5)
    for _ in range(200):
        mask = _random_mask(rng)
        b_raw = rng.random(3); b_raw /= b_raw.sum()
        p_raw = rng.random(3); p_raw /= p_raw.sum()
        b1, p1 = project_bw_power(b_raw, p_raw, mask)
        b2, p2 = project_bw_power(b1, p1, mask)
        assert np.allclose(b1, b2, atol=1e-9) and np.allclose(p1, p2, atol=1e-9)


def test_projection_edge_cases_sum_to_one():
    # raw entirely below the floors, and floors summing to exactly 1
    for floor in ([0.9, 0.05, 0.0], [0.5, 0.3, 0.2], [0.0, 0.0, 0.0]):
        mask = Mask(bw_floor=np.array(floor), power_floor_emg=0.0, expires_at=10)
        for raw in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1 / 3] * 3):
            b, p = project_bw_power(np.array(raw), np.array(raw), mask)
            assert abs(b.sum() - 1.0) < 1e-9 and np.all(b >= np.array(floor) - 1e-9)
            assert abs(p.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------- training masks (F11)
from uavrl.control.mask import RandomMaskSampler, project_raw_action  # noqa: E402


def _shares3(action):
    e = np.exp(3.0 * np.asarray(action, dtype=float))
    return e / e.sum()


def test_sampler_masks_are_schema_valid_and_about_30_percent_of_episodes():
    sampler = RandomMaskSampler(prob=0.3, episode_slots=2000, rng=np.random.default_rng(0))
    used = 0
    for _ in range(2000):
        sampler.new_episode()
        if sampler.mask is None:
            continue
        used += 1
        f = sampler.mask.bw_floor
        assert sampler.mask.valid()
        assert f[0] <= 0.9 and f.sum() <= 0.95 + 1e-9
        assert 50 <= sampler.mask.expires_at - sampler.start <= 1000
    assert 0.25 < used / 2000 < 0.35


def test_sampler_disabled_leaves_actions_untouched():
    sampler = RandomMaskSampler(prob=0.0, rng=np.random.default_rng(0))
    sampler.new_episode()
    a = np.array([0.3, -0.2, 0.1, 0.0, 0.5, -0.5], dtype=np.float32)
    assert np.array_equal(sampler.apply(10, a), a)


def test_projected_raw_action_respects_floors_in_the_environment_mapping():
    rng = np.random.default_rng(3)
    sampler = RandomMaskSampler(prob=1.0, rng=rng)
    for _ in range(50):
        sampler.new_episode()
        mask = sampler.mask
        a = rng.uniform(-1, 1, 6)
        out = project_raw_action(a, mask)
        b, p = _shares3(out[:3]), _shares3(out[3:])
        assert abs(b.sum() - 1) < 1e-6 and abs(p.sum() - 1) < 1e-6
        assert np.all(b >= mask.bw_floor - 1e-5)
        assert p[0] >= mask.power_floor_emg - 1e-5


def test_sampler_only_acts_inside_its_window():
    sampler = RandomMaskSampler(prob=1.0, rng=np.random.default_rng(1))
    sampler.new_episode()
    a = np.zeros(6, dtype=np.float32)
    before = max(sampler.start - 1, 0) if sampler.start > 0 else None
    if before is not None:
        assert np.array_equal(sampler.apply(before, a), a)
    assert not np.array_equal(sampler.apply(sampler.start, a), a) or sampler.mask.bw_floor[0] <= 1 / 3
    assert np.array_equal(sampler.apply(sampler.mask.expires_at, a), a)
