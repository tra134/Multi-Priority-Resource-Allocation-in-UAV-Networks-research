import numpy as np
import pytest

from uavmec.config import LinConfig
from uavmec.env.lin_env import LinMECEnv
from uavmec.env.types import SensorAction


def _rand_actions(env, rng, fos=False):
    cfg = env.cfg
    lo = 1 if fos else 0
    return [
        SensorAction(
            int(rng.integers(lo, cfg.n_uavs + 1)),
            int(rng.integers(cfg.n_subcarriers)),
            float(rng.uniform()),
            float(rng.uniform()),
        )
        for _ in range(cfg.n_sensors)
    ]


def _run(env, seed, fos=False):
    rng = np.random.default_rng(123)
    obs, _ = env.reset(seed)
    traj = [np.concatenate(obs)]
    rewards = []
    done = False
    while not done:
        obs, r, _, trunc, _ = env.step(_rand_actions(env, rng, fos))
        traj.append(np.concatenate(obs))
        rewards.append(r)
        done = trunc
    return np.array(traj), np.array(rewards)


def test_shapes_and_attributes():
    cfg = LinConfig()
    env = LinMECEnv(cfg)
    obs, info = env.reset(0)
    assert len(obs) == env.n_agents == 8
    assert all(o.shape == (env.obs_dim,) for o in obs)
    assert info["state"].shape == (env.state_dim,)
    assert env.sensor_pos.shape == (8, 2) and env.uav_pos.shape == (2, 2)
    assert env.task_sizes.shape == env.task_deadlines.shape == env.battery.shape == (8,)
    assert env.uav_battery.shape == (2,) and env.channel_gain.shape == (8, 2)
    assert env.t == 0


def test_observation_normalized():
    env = LinMECEnv(LinConfig())
    obs, _ = env.reset(3)
    for o in obs:
        assert o.min() >= 0.0 and o.max() <= 1.0 + 1e-6


def test_determinism_and_different_seeds():
    env = LinMECEnv(LinConfig())
    a, ra = _run(env, 7)
    b, rb = _run(env, 7)
    c, _ = _run(env, 8)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)
    assert not np.array_equal(a, c)


def test_truncation_at_T_and_rewards():
    cfg = LinConfig()
    env = LinMECEnv(cfg)
    rng = np.random.default_rng(0)
    env.reset(0)
    for t in range(cfg.n_slots):
        _, r, term, trunc, info = env.step(_rand_actions(env, rng))
        assert term is False
        assert trunc == (t == cfg.n_slots - 1)
        assert set(np.unique(r)).issubset({-1.0, 3.0})
        assert r.shape == (8,)
        assert len(info["outcomes"]) == 8 and info["uav_energy_j"].shape == (2,)
        for rew, o in zip(r, info["outcomes"]):
            assert rew == (3.0 if o.success else -1.0)
            assert (o.fail_reason is None) == o.success
    assert env.t == cfg.n_slots


def test_fos_rejects_target_zero_and_pos_accepts():
    env = LinMECEnv(LinConfig(), mode="FOS")
    env.reset(0)
    acts = [SensorAction(1)] * 7 + [SensorAction(0)]
    with pytest.raises(ValueError):
        env.step(acts)
    traj, _ = _run(env, 1, fos=True)
    assert len(traj) == LinConfig().n_slots + 1
    pos = LinMECEnv(LinConfig(), mode="POS")
    pos.reset(0)
    pos.step([SensorAction(0)] * 8)


def test_invalid_actions():
    env = LinMECEnv(LinConfig())
    env.reset(0)
    with pytest.raises(ValueError):
        env.step([SensorAction(3)] * 8)
    with pytest.raises(ValueError):
        env.step([SensorAction(1, subcarrier=4)] * 8)
    with pytest.raises(ValueError):
        env.step([SensorAction(1, power_frac=1.5)] * 8)
    with pytest.raises(ValueError):
        env.step([SensorAction(1)] * 7)


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 12])
def test_sensor_counts_run(n):
    env = LinMECEnv(LinConfig(n_sensors=n))
    _, rewards = _run(env, 0)
    assert rewards.shape == (10, n)
    assert env.obs_dim == 15


def test_energy_accounting():
    cfg = LinConfig()
    env = LinMECEnv(cfg)
    rng = np.random.default_rng(4)
    env.reset(2)
    for _ in range(cfg.n_slots):
        before = env.battery.copy()
        _, _, _, _, info = env.step(_rand_actions(env, rng))
        spent = np.array([o.energy_j for o in info["outcomes"]])
        assert np.allclose(before - env.battery, spent)
        assert (spent >= 0).all()


def test_local_outcome_values():
    cfg = LinConfig()
    env = LinMECEnv(cfg)
    env.reset(0)
    bits = env.task_sizes.copy()
    _, _, _, _, info = env.step([SensorAction(0)] * 8)
    for o, b in zip(info["outcomes"], bits):
        assert o.mode == "local"
        assert np.isclose(o.latency_s, b * cfg.c_cyc / 1e9)
        assert np.isclose(o.energy_j, b * cfg.c_cyc * cfg.kappa * 1e18)
        assert o.success == (o.latency_s <= o.deadline_s)


def test_sic_beats_nosic_on_shared_subcarrier():
    acts = [SensorAction(1, 0, 1.0, 1.0)] * 8
    succ = {}
    for sic in (True, False):
        env = LinMECEnv(LinConfig(), sic=sic)
        env.reset(0)
        _, _, _, _, info = env.step(acts)
        succ[sic] = sum(o.success for o in info["outcomes"])
    assert succ[True] >= succ[False]


def test_uavs_hover_and_no_mobility_option():
    from uavmec.config import load_config

    assert not hasattr(load_config(), "uav_mobility")  # A-13
    env = LinMECEnv(LinConfig())
    env.reset(0)
    start = env.uav_pos.copy()
    done = False
    while not done:
        _, _, _, done, info = env.step([SensorAction(0)] * 8)
        assert np.array_equal(env.uav_pos, start)
    assert (info["uav_energy_j"] > 0).all()


def test_uav_min_separation():
    env = LinMECEnv(LinConfig())
    for s in range(20):
        env.reset(s)
        assert np.linalg.norm(env.uav_pos[0] - env.uav_pos[1]) >= env.cfg.d_safe_m
