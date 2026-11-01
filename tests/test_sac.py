"""Tests for the SAC / REDQ-SAC agent (plan.md Phases 3-4).

- select_action returns a 6-dim action in [-1, 1].
- update() runs without error and produces finite losses.
- uncertainty() returns a finite std/mean over the critic ensemble.
- N=1 (single critic) behaves like vanilla SAC; N>1 gives ensemble.
"""

from __future__ import annotations

import numpy as np

from uavrl.agents.replay import ReplayBuffer
from uavrl.agents.sac import SACAgent


def make_agent(N_critics=10, seed=0):
    rng = np.random.default_rng(seed)
    return SACAgent(
        obs_dim=17,
        act_dim=6,
        device="cpu",
        seed=seed,
        N_critics=N_critics,
        rng=rng,
    )


def test_action_range():
    agent = make_agent(N_critics=2)
    obs = np.zeros(17, dtype=np.float32)
    a = agent.select_action(obs, deterministic=False)
    assert a.shape == (6,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    ad = agent.select_action(obs, deterministic=True)
    assert np.all(ad >= -1.0) and np.all(ad <= 1.0)


def test_update_runs():
    agent = make_agent(N_critics=10, seed=1)
    buf = ReplayBuffer(17, 6, size=1000, rng=np.random.default_rng(2))
    rng = np.random.default_rng(3)
    for _ in range(400):
        obs = rng.normal(size=17).astype(np.float32)
        act = rng.uniform(-1, 1, 6).astype(np.float32)
        buf.store(obs, act, float(rng.normal()), rng.normal(size=17).astype(np.float32), 0.0)
    logs = agent.update(buf, batch_size=64)
    assert np.isfinite(logs["critic_loss"])
    assert np.isfinite(logs["actor_loss"])
    assert np.isfinite(logs["alpha"]) and logs["alpha"] > 0


def test_uncertainty_finite():
    agent = make_agent(N_critics=10, seed=4)
    obs = np.zeros(17, dtype=np.float32)
    act = np.zeros(6, dtype=np.float32)
    u, m = agent.uncertainty(obs, act)
    assert np.isfinite(u) and np.isfinite(m)
    assert u >= 0.0


def test_single_critic_uncertainty_zero():
    # A single-critic ensemble has no disagreement -> std = 0.
    agent = make_agent(N_critics=1, seed=5)
    obs = np.ones(17, dtype=np.float32) * 0.5
    act = np.ones(6, dtype=np.float32) * 0.2
    u, _ = agent.uncertainty(obs, act)
    assert u < 1e-6


def test_save_load(tmp_path):
    agent = make_agent(N_critics=5, seed=7)
    path = tmp_path / "sac_test.pt"
    agent.save(str(path))
    agent2 = make_agent(N_critics=5, seed=99)
    agent2.load(str(path), device="cpu")
    obs = np.zeros(17, dtype=np.float32)
    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent2.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2)


def test_checkpoint_is_resumable(tmp_path):
    """Optimizer state and the training step survive a save/load round-trip."""
    agent = make_agent(N_critics=5, seed=8)
    buf = ReplayBuffer(17, 6, size=1000, rng=np.random.default_rng(9))
    rng = np.random.default_rng(10)
    for _ in range(320):
        obs = rng.normal(size=17).astype(np.float32)
        act = rng.uniform(-1, 1, 6).astype(np.float32)
        buf.store(obs, act, float(rng.normal()), rng.normal(size=17).astype(np.float32), 0.0)
    agent.update(buf, batch_size=64)
    agent.total_it = 1

    path = tmp_path / "resume.pt"
    agent.save(str(path), step=1234)

    fresh = make_agent(N_critics=5, seed=11)
    payload = fresh.load(str(path), device="cpu")

    assert payload["step"] == 1234
    assert fresh.total_it == agent.total_it
    # Optimizer state is restored (Adam has exp_avg for the actor).
    assert fresh.actor_optimizer.state_dict()["state"]
    assert fresh.critic_optimizer.state_dict()["state"]
    assert fresh.alpha_optimizer.state_dict()["state"]

    obs = np.zeros(17, dtype=np.float32)
    assert np.allclose(
        agent.select_action(obs, deterministic=True),
        fresh.select_action(obs, deterministic=True),
    )
