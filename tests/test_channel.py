"""Tests for the channel and traffic models (plan.md Phase 1).

- p_los in (0,1), monotone increasing in elevation angle.
- Path loss at 1 m equals the free-space formula within 1e-6 dB.
- Rate increases with b and with p; zero when b=0 or p=0.
- Over a long run, empirical LoS fraction within 3% of theoretical mean.
- Poisson arrivals have the correct mean; rate-multiplier hook scales arrivals.
"""

from __future__ import annotations

import numpy as np

from uavrl.env.channel import (
    SPEED_OF_LIGHT,
    free_space_path_loss_db,
    noise_power_w,
    p_los,
    rate,
    Channel,
)
from uavrl.env.traffic import Traffic, TrafficSource


def test_p_los_range_and_monotone():
    thetas = np.linspace(0.0, 90.0, 200)
    p = p_los(thetas)
    assert np.all(p > 0.0) and np.all(p < 1.0)
    assert np.all(np.diff(p) > 0.0), "p_los must be monotone increasing"


def test_path_loss_at_1m_matches_free_space():
    fc = 2.0e9
    expected = 20.0 * np.log10(4.0 * np.pi * 1.0 * fc / SPEED_OF_LIGHT)
    assert abs(free_space_path_loss_db(1.0, fc) - expected) < 1e-6


def test_rate_monotone_and_zero():
    n0 = noise_power_w(1.0, -174.0, 7.0)
    g = 1e-3
    # rate increases with bandwidth
    rb = rate(np.array([1.0e6, 2.0e6]), np.array([0.1, 0.1]), np.array([g, g]), n0)
    assert rb[1] > rb[0]
    # rate increases with power
    rp = rate(np.array([1.0e6, 1.0e6]), np.array([0.1, 0.2]), np.array([g, g]), n0)
    assert rp[1] > rp[0]
    # zero when b=0 or p=0
    assert rate(0.0, 0.1, g, n0) == 0.0
    assert rate(1.0e6, 0.0, g, n0) == 0.0


def test_empirical_los_fraction_within_3pct():
    rng = np.random.default_rng(0)
    ch = Channel(n_classes=1, H=100.0, fc=2e9, los_stay_prob=0.98, rng=rng)
    radius = np.array([100.0])
    p_theory = float(p_los(ch.elevation_angle_deg(radius)[0]))
    n = 20000
    count = 0
    for _ in range(n):
        ch.update(radius, shadow_db=0.0)
        count += int(ch.los_state[0])
    empirical = count / n
    assert abs(empirical - p_theory) < 0.03


def test_poisson_arrivals_mean():
    rng = np.random.default_rng(1)
    src = TrafficSource(mean_mbps=1.0, packet_bytes=200, rng=rng)
    dt = 0.005
    samples = np.array([src.arrivals_bits(dt) for _ in range(50000)])
    expected = 1.0e6 * dt  # 1 Mbps * 5 ms = 5000 bits
    assert abs(samples.mean() - expected) / expected < 0.02


def test_rate_multiplier_hook():
    rng = np.random.default_rng(2)
    src = TrafficSource(mean_mbps=1.0, packet_bytes=200, rng=rng)
    dt = 0.005
    base = src.arrivals_bits(dt, multiplier=1.0)
    surged = src.arrivals_bits(dt, multiplier=4.0)
    # Surge uses a 4x Poisson rate; in expectation 4x the arrivals.  We check
    # statistically over many samples instead of a single draw.
    samples_base = np.mean([src.arrivals_bits(dt, 1.0) for _ in range(20000)])
    samples_surge = np.mean([src.arrivals_bits(dt, 4.0) for _ in range(20000)])
    assert samples_surge > samples_base * 3.0


def test_traffic_per_class():
    rng = np.random.default_rng(3)
    tr = Traffic([0.5, 8.0, 6.0], packet_bytes=200, rng=rng)
    a = tr.arrivals_bits(0.005, multipliers=[4.0, 1.0, 1.0])
    assert a.shape == (3,)
    assert np.all(a >= 0)
