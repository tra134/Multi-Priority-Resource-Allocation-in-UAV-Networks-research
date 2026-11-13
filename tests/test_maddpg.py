import numpy as np

from uavmec.agents.maddpg import MADDPG
from uavmec.config import load_config
from uavmec.env.lin_env import LinMECEnv

CFG = {
    "batch": 16, "actor_lr": 3e-4, "critic_lr": 5e-5, "buffer": 1000, "gamma": 0.9,
    "tau_soft": 0.01, "warmup": 32, "gumbel_tau": 1.0, "noise_sigma_start": 0.1,
    "noise_sigma_end": 0.01, "noise_decay_steps": 100, "actor_hidden": 32,
    "critic_hidden": [16, 8],
}


def _make(fos=False, seed=0):
    env = LinMECEnv(load_config(n_sensors=2), mode="FOS" if fos else "POS")
    ag = MADDPG(env.n_agents, env.obs_dim, env.state_dim, env.cfg.n_uavs, env.cfg.n_subcarriers,
                CFG, fos, seed)
    return env, ag


def _run(env, ag, steps, obs, state):
    losses = []
    for _ in range(steps):
        acts = ag.act(obs, state, True)
        nobs, rew, _, trunc, info = env.step(acts)
        ag.observe(obs, state, rew, nobs, info["state"], trunc)
        losses.append(ag.update())
        obs, state = nobs, info["state"]
        if trunc:
            obs, info = env.reset()
            state = info["state"]
    return obs, state, losses


def test_runs_and_losses_finite():
    env, ag = _make(fos=True)
    obs, info = env.reset(seed=1)
    _, _, losses = _run(env, ag, 200, obs, info["state"])
    assert losses[10] == {}
    done = [x for x in losses if x]
    assert done and all(np.isfinite(v) for x in done for v in x.values())


def test_state_dict_resume_determinism():
    env, ag = _make(seed=3)
    obs, info = env.reset(seed=1)
    obs, state, _ = _run(env, ag, 60, obs, info["state"])
    sd = ag.state_dict()
    _, ag2 = _make(seed=99)
    ag2.load_state_dict(sd)
    assert ag.act(obs, state, True) == ag2.act(obs, state, True)
    assert ag.update() == ag2.update()
