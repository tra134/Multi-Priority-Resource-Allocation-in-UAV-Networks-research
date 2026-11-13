import numpy as np

from uavmec.config import LinConfig
from uavmec.env.tasks import sample_tasks


def test_ranges_and_shapes():
    cfg = LinConfig()
    bits, ddl = sample_tasks(np.random.default_rng(0), 2000, cfg)
    assert bits.shape == ddl.shape == (2000,)
    assert bits.min() >= 100e3 and bits.max() <= 1.5e6
    assert ddl.min() >= 0.001 and ddl.max() <= 0.050
    # sizes are multiples of C = 0.05 Mbit (A-5)
    assert np.allclose(bits / 50e3, np.round(bits / 50e3))


def test_mean_size_and_deadline_independent_uniform():
    cfg = LinConfig()
    bits, ddl = sample_tasks(np.random.default_rng(1), 20000, cfg)
    assert abs(bits.mean() - 0.8e6) < 0.02e6  # lambda = 16 -> 0.8 Mbit (A-5)
    # A-33: D_n ~ U(1, 50) ms, independent of C_n (test T-1, replaced A-6)
    assert abs(ddl.mean() - 0.0255) < 0.0005
    assert abs(np.corrcoef(bits, ddl)[0, 1]) < 0.03


def test_deterministic():
    cfg = LinConfig()
    a = sample_tasks(np.random.default_rng(5), 8, cfg)
    b = sample_tasks(np.random.default_rng(5), 8, cfg)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
