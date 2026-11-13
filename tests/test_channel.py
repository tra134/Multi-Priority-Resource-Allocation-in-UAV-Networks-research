import numpy as np

from uavmec.env.channel import channel_gain


def test_gain_hand_computed():
    # sensor (0,0), UAV (30,40): d^2 = 2500, H^2 = 10000 -> 1e-5 / 12500 = 8e-10
    g = channel_gain(np.array([[0.0, 0.0]]), np.array([[30.0, 40.0]]), 100.0, 1e-5)
    assert g.shape == (1, 1)
    assert np.isclose(g[0, 0], 8e-10)


def test_gain_directly_below_and_shape():
    g = channel_gain(np.zeros((3, 2)), np.zeros((2, 2)), 100.0, 1e-5)
    assert g.shape == (3, 2)
    assert np.allclose(g, 1e-9)  # -50 dB at 100 m height, 1e-5 / 1e4
