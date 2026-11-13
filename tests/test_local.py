import numpy as np

from uavmec.env.local import local_latency


def test_eq1_hand_computed():
    # 0.8 Mbit * 23 cycles/bit / 1 GHz = 18.4 ms
    assert np.isclose(local_latency(0.8e6, 23.0, 1e9), 0.0184)


def test_eq1_vector():
    out = local_latency(np.array([1e6, 2e6]), 10.0, 1e9)
    assert np.allclose(out, [0.01, 0.02])
