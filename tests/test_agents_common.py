import numpy as np
import torch

from uavmec.agents.common import ACTION_DIM, Actor, decode_actions


def test_decode_shapes_and_fos_never_local():
    K, M, N = 3, 4, 5
    raw = np.zeros((N, ACTION_DIM(K, M)))
    raw[:, 0] = 1.0  # strongly prefers local
    assert all(a.target == 0 for a in decode_actions(raw, K, M, fos=False))
    acts = decode_actions(raw, K, M, fos=True)
    assert len(acts) == N and all(1 <= a.target <= K for a in acts)
    assert all(0 <= a.subcarrier < M and 0 <= a.power_frac <= 1 for a in acts)


def test_actor_ranges_and_gradient():
    K, M = 2, 3
    actor = Actor(7, K, M)
    obs = torch.randn(6, 7)
    for explore in (True, False):
        out = actor(obs, explore, 1.0, fos=True)
        assert out.shape == (6, ACTION_DIM(K, M))
        assert torch.all(out[:, 0] == 0)
        assert torch.allclose(out[:, : K + 1].sum(1), torch.ones(6))
        assert torch.allclose(out[:, K + 1 : K + 1 + M].sum(1), torch.ones(6))
        assert torch.all((out[:, -2:] >= 0) & (out[:, -2:] <= 1))
    out = actor(obs, True, 1.0, False)
    (out[:, : K + 1] * torch.randn(K + 1)).sum().backward()
    assert actor.target_head.weight.grad.abs().sum() > 0
