"""Tests for the MAPPO learner (spec Section 10)."""

import copy

import numpy as np

from uavmec.agents.mappo import MAPPO
from uavmec.config import load_config
from uavmec.env.lin_env import LinMECEnv

CFG = {"actor_lr": 3e-4, "critic_lr": 5e-5, "gamma": 0.9, "gae_lambda": 0.95, "clip": 0.2,
       "epochs": 2, "minibatch": 8, "rollout_episodes": 2, "actor_hidden": 16,
       "critic_hidden": [16, 8], "ent_coef": 0.01}


def _make(mode="POS", seed=0):
    env = LinMECEnv(load_config(n_sensors=2), mode=mode)
    agent = MAPPO(env.n_agents, env.obs_dim, env.state_dim, env.cfg.n_uavs,
                  env.cfg.n_subcarriers, CFG, mode == "FOS", seed)
    return env, agent


def _episode(env, agent, seed, explore=True):
    obs, info = env.reset(seed=seed)
    state, losses, acts, done = info["state"], [], [], False
    while not done:
        a = agent.act(obs, state, explore)
        acts.append(a)
        nobs, r, _, trunc, info = env.step(a)
        done = bool(trunc)
        agent.observe(obs, state, r, nobs, info["state"], done)
        losses.append(agent.update())
        obs, state = nobs, info["state"]
    return acts, losses


def test_fos_never_local_and_actions_valid():
    env, agent = _make("FOS")
    for explore in (True, False):
        acts, _ = _episode(env, agent, 1, explore)  # env validates every action
        assert all(a.target >= 1 for step in acts for a in step)


def test_update_after_two_episodes_with_finite_losses():
    env, agent = _make()
    _, l1 = _episode(env, agent, 1)
    assert all(x == {} for x in l1)
    _, l2 = _episode(env, agent, 2)
    assert all(x == {} for x in l2[:-1]) and l2[-1]
    assert all(np.isfinite(v) for v in l2[-1].values())
    assert agent.rollout == [] and agent.n_updates == 1


def test_state_dict_roundtrip():
    env, agent = _make(seed=3)
    _episode(env, agent, 1)
    snap = copy.deepcopy(agent.state_dict())
    acts_a, losses_a = _episode(env, agent, 2)
    _, clone = _make(seed=99)
    clone.load_state_dict(snap)
    acts_b, losses_b = _episode(env, clone, 2)
    assert acts_a == acts_b and losses_a == losses_b
