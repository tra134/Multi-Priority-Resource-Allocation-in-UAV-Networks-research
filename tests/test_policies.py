import pytest

pytest.importorskip("uavmec.env.lin_env")

from uavmec.env.lin_env import LinMECEnv

from uavmec.baselines.policies import (
    GreedyLocalOrOffload,
    LocalComputing,
    NearestUAVFullPower,
    RandomPolicy,
)
from uavmec.config import load_config


def make(mode="POS"):
    return LinMECEnv(load_config(), sic=True, mode=mode)


def run(env, pol, seed=1):
    import numpy as np

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    outs = []
    while True:
        obs, _, term, trunc, info = env.step(pol.act(env, obs, rng))
        outs += info["outcomes"]
        if term or trunc:
            return outs


def test_local_only_local():
    assert {o.mode for o in run(make(), LocalComputing())} == {"local"}


def test_fos_random_never_local():
    env = make("FOS")
    assert all(o.mode == "offload" for o in run(env, RandomPolicy()))
    obs, _ = env.reset(seed=3)
    import numpy as np

    acts = RandomPolicy().act(env, obs, np.random.default_rng(0))
    assert all(a.target >= 1 for a in acts)


def test_nearest_targets_valid():
    env = make()
    obs, _ = env.reset(seed=4)
    import numpy as np

    for pol in (NearestUAVFullPower(), GreedyLocalOrOffload()):
        acts = pol.act(env, obs, np.random.default_rng(0))
        assert len(acts) == env.n_agents
        assert all(0 <= a.target <= len(env.uav_pos) for a in acts)
