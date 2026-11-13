import numpy as np

from uavmec.env.noma import noma_group, noma_links

NOISE = 1e-10
G = np.array([4e-9, 2e-9, 1e-9])
BW = np.full(3, 1e6)


def _rate(bw, sinr):
    return bw * np.log2(1.0 + sinr)


def test_sic_all_decode_removes_interference():
    sinr, rate, dec = noma_group(G, BW, NOISE, 1.0, sic=True)
    # strongest sees both others, then user 1 sees only user 2, user 2 sees noise only
    assert np.allclose(sinr, [4e-9 / (3e-9 + NOISE), 2e-9 / (1e-9 + NOISE), 1e-9 / NOISE])
    assert np.allclose(rate, _rate(1e6, sinr))
    assert dec.all()


def test_decode_order_independent_of_input_order():
    perm = np.array([2, 0, 1])
    sinr, _, _ = noma_group(G[perm], BW, NOISE, 1.0, sic=True)
    ref, _, _ = noma_group(G, BW, NOISE, 1.0, sic=True)
    assert np.allclose(sinr, ref[perm])


def test_failed_decode_stays_as_interference():
    # user 0 rate = 1e6*log2(1+1.2903) = 1.196e6 < 1.3e6 -> fails; user 1 then sees 4e-9 + 1e-9
    sinr, _, dec = noma_group(G, BW, NOISE, 1.3e6, sic=True)
    assert not dec[0]
    assert np.isclose(sinr[1], 2e-9 / (5e-9 + NOISE))
    assert not dec[1]  # rate 1e6*log2(1.392) = 0.47e6
    assert np.isclose(sinr[2], 1e-9 / (6e-9 + NOISE))  # both failed users interfere


def test_nosic_sums_all_interference():
    sinr, _, dec = noma_group(G, BW, NOISE, 1.0, sic=False)
    total = G.sum()
    assert np.allclose(sinr, G / (total - G + NOISE))
    assert dec.all()


def test_single_user_snr_and_rate_threshold():
    sinr, rate, dec = noma_group(np.array([1e-9]), np.array([250e6]), NOISE, 10e6)
    assert np.isclose(sinr[0], 10.0)
    assert np.isclose(rate[0], 250e6 * np.log2(11.0))
    assert dec[0]
    _, _, dec2 = noma_group(np.array([1e-9]), np.array([1.0]), NOISE, 10e6)
    assert not dec2[0]


def test_zero_power_user_not_decoded():
    sinr, rate, dec = noma_group(np.array([0.0]), np.array([1e6]), NOISE, 10e6)
    assert sinr[0] == 0 and rate[0] == 0 and not dec[0]


def test_links_groups_are_independent():
    uav = np.array([0, 0, 1])
    sub = np.array([0, 1, 0])
    rx = np.array([4e-9, 2e-9, 1e-9])
    sinr, _, _ = noma_links(uav, sub, rx, np.full(3, 1e6), NOISE, 1.0, True)
    assert np.allclose(sinr, rx / NOISE)  # nobody shares a (UAV, subcarrier)
    sub2 = np.array([0, 0, 0])
    sinr2, _, _ = noma_links(uav, sub2, rx, np.full(3, 1e6), NOISE, 1.0, True)
    assert np.isclose(sinr2[0], 4e-9 / (2e-9 + NOISE))  # users 0 and 1 share (0,0)
    assert np.isclose(sinr2[2], 1e-9 / NOISE)  # alone on UAV 1
