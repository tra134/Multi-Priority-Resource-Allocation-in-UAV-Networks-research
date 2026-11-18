"""Regression tests for the S-3 sensitivity variants (docs/lin2025_spec.md, Section 11).

Variants are selected only through LinMECEnv(..., s3_variant=...); the default path is v1.0.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from uavmec.baselines.policies import POLICIES
from uavmec.config import load_config
from uavmec.env.lin_env import LinMECEnv
from uavmec.env.noma import noma_group
from uavmec.env.s3_variants import (
    S3Variant,
    allocate_bandwidth,
    load_s3_variants,
    noma_group_a8lit,
)
from uavmec.env.types import SensorAction
from uavmec.eval.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "tests" / "data" / "v1_reference_s3.json").read_text())
SEEDS = range(2000, 2005)
ALL = ["V0", "N1-seq", "N1-prop", "N1-eq", "A1-const", "A8-lit", "A3-total", "A2-dBm", "A2-psd", "UAV-cb"]
COLS = ("success_rate", "mean_latency_ms", "mean_energy_mw", "n_local", "n_offload",
        "fail_deadline", "fail_energy", "fail_rate", "fail_decode", "n_tasks", "n_success")
B = 250.0e6
NOISE = 1e-10


def variants():
    return load_s3_variants(ROOT / "configs" / "s3_variants.yaml")


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def episodes(variant, policy, n, sic=True):
    cfg = load_config(ROOT / "configs" / "lin2025.yaml", n_sensors=n)
    env = LinMECEnv(cfg, sic=sic, mode="POS", s3_variant=variant)
    return [run_episode(env, POLICIES[policy](), s) for s in SEEDS]


# ------------------------------------------------------------ config file
def test_config_has_all_variants_and_unknown_key_rejected():
    v = variants()
    assert list(v) == ALL
    assert all(isinstance(x, S3Variant) for x in v.values())
    with pytest.raises(ValueError):
        S3Variant.from_dict("bad", {"not_a_key": 1})
    with pytest.raises(ValueError):
        S3Variant.from_dict("bad", {"bw_rule": "nope"})


def test_default_env_has_no_variant():
    assert LinMECEnv(load_config(ROOT / "configs" / "lin2025.yaml")).s3 is None


# ------------------------------------------------- V0 == v1.0 (regression)
@pytest.mark.parametrize("n", [2, 8, 12])
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_v0_and_default_match_frozen_v1_reference(policy, n):
    ref = REF[f"{policy}|{n}"]
    for variant in (None, variants()["V0"]):
        got = episodes(variant, policy, n)
        for r, g in zip(ref, got, strict=True):
            assert r["seed"] == g["seed"]
            for c in COLS:
                assert same(r[c], g[c]), (variant, policy, n, g["seed"], c, r[c], g[c])


# ------------------------------------------------------ N-1 bandwidth rules
def test_n1_formulas_on_hand_built_group():
    f = np.array([0.5, 0.6, 0.4])
    sid, k, m = np.array([0, 1, 2]), np.zeros(3, int), np.zeros(3, int)
    seq = allocate_bandwidth("seq", sid, k, m, f, B)
    assert np.allclose(seq, B * np.array([0.5, 0.6 * 0.5, 0.4 * 0.2]))  # R: 1 -> .5 -> .2
    prop = allocate_bandwidth("prop", sid, k, m, f, B)
    assert np.allclose(prop, B * f / 1.5)
    eq = allocate_bandwidth("eq", sid, k, m, f, B)
    assert np.allclose(eq, B / 3)
    v0 = allocate_bandwidth("v0", sid, k, m, f, B)
    assert np.allclose(v0, f * B)


def test_n1_prop_does_not_rescale_when_sum_at_most_one():
    f = np.array([0.3, 0.4])
    out = allocate_bandwidth("prop", np.array([0, 1]), np.zeros(2, int), np.zeros(2, int), f, B)
    assert np.allclose(out, f * B)


def test_n1_equivalence_when_all_fractions_one():
    f = np.ones(3)
    sid, k, m = np.array([0, 1, 2]), np.zeros(3, int), np.zeros(3, int)
    prop = allocate_bandwidth("prop", sid, k, m, f, B)
    eq = allocate_bandwidth("eq", sid, k, m, f, B)
    seq = allocate_bandwidth("seq", sid, k, m, f, B)
    assert np.array_equal(prop, eq) and np.allclose(prop, B / 3)
    assert np.allclose(seq, [B, 0.0, 0.0])


def test_n1_groups_are_independent_and_seq_uses_sensor_index_order():
    # sensors 5,2,9 on (uav0,sub0); sensor 7 alone on (uav0,sub1); sensor 3 alone on (uav1,sub0)
    sid = np.array([5, 2, 9, 7, 3])
    k = np.array([0, 0, 0, 0, 1])
    m = np.array([0, 0, 0, 1, 0])
    f = np.array([0.5, 0.5, 0.5, 0.9, 0.7])
    seq = allocate_bandwidth("seq", sid, k, m, f, B)
    # in the (0,0) group sensor 2 goes first, then 5, then 9
    assert np.allclose(seq, B * np.array([0.25, 0.5, 0.125, 0.9, 0.7]))
    eq = allocate_bandwidth("eq", sid, k, m, f, B)
    assert np.allclose(eq, B * np.array([1 / 3, 1 / 3, 1 / 3, 1.0, 1.0]))


# --------------------------------------------------------------- A-8 literal
H = np.array([1e-9, 2e-9, 4e-9])
BW = np.full(3, 1e6)


def _r(bw, gamma):
    return bw * np.log2(1.0 + gamma)


def test_a8lit_sic_hand_computed_distinct_powers_all_decode():
    p = np.array([0.2, 1.0, 0.5])  # decode order: sensor 1, 2, 0
    sinr, rate, dec = noma_group_a8lit(H, p, BW, NOISE, 1.0, sic=True)
    g1 = 2e-9 * 1.0 / (2e-9 * (0.2 + 0.5) + 1e-10)  # nobody cancelled
    g2 = 4e-9 * 0.5 / (4e-9 * 0.2 + 1e-10)  # sensor 1 cancelled, sensor 0 not
    g0 = 1e-9 * 0.2 / 1e-10  # sensors 1 and 2 cancelled
    assert np.allclose(sinr, [g0, g1, g2])
    assert np.allclose(rate, _r(1e6, np.array([g0, g1, g2])))
    assert dec.all()


def test_a8lit_sic_failed_strong_user_stays_as_interference():
    p = np.array([0.2, 1.0, 0.5])
    sinr, _, dec = noma_group_a8lit(H, p, BW, NOISE, 1.3e6, sic=True)
    g1 = 2e-9 / (1.4e-9 + 1e-10)  # = 1.3333 -> rate 1.222e6 < 1.3e6: fails
    g2 = 2e-9 / (4e-9 * 1.2 + 1e-10)  # sensor 1 failed: stays (4e-9 * 1.0) plus sensor 0 (4e-9 * 0.2)
    g0 = 2e-10 / (1e-9 * 1.5 + 1e-10)  # sensors 1 and 2 both failed and stay
    assert np.allclose(sinr, [g0, g1, g2])
    assert not dec.any()


def test_a8lit_equal_powers_nothing_is_cancelled():
    p = np.ones(3)
    sinr, _, dec = noma_group_a8lit(H, p, BW, NOISE, 1.0, sic=True)
    assert np.allclose(sinr, H / (2 * H + NOISE))
    sinr2, _, _ = noma_group_a8lit(H, p, BW, NOISE, 1.0, sic=False)
    assert np.allclose(sinr, sinr2)
    assert dec.all()


def test_a8lit_nosic_uses_own_gain_for_every_interferer():
    p = np.array([0.2, 1.0, 0.5])
    sinr, _, _ = noma_group_a8lit(H, p, BW, NOISE, 1.0, sic=False)
    expect = H * p / (H * (p.sum() - p) + NOISE)
    assert np.allclose(sinr, expect)
    # differs from V0 noSIC, which uses sum_j h_j p_j
    v0, _, _ = noma_group(H * p, BW, NOISE, 1.0, sic=False)
    assert not np.allclose(sinr, v0)


def test_a8lit_single_user_is_snr():
    sinr, _, _ = noma_group_a8lit(H[:1], np.array([0.7]), BW[:1], NOISE, 1.0, sic=True)
    assert np.allclose(sinr, H[0] * 0.7 / NOISE)


# ------------------------------------------------------ env-level formulas
def _env(variant_name, n=2, sic=True, mode="POS"):
    cfg = load_config(ROOT / "configs" / "lin2025.yaml", n_sensors=n)
    v = None if variant_name is None else variants()[variant_name]
    return LinMECEnv(cfg, sic=sic, mode=mode, s3_variant=v)


def _first_step(name, actions, n=2, seed=2000, sic=True):
    env = _env(name, n, sic)
    env.reset(seed)
    gains, bits = env.channel_gain.copy(), env.task_sizes.copy()
    _, _, _, _, info = env.step(actions)
    return info["outcomes"], gains, bits


LONE = [SensorAction(1, 0, 1.0, 1.0), SensorAction(0, 0, 0.0, 0.0)]


def test_a1_const_gain_is_1e_minus_5_on_every_link():
    env = _env("A1-const", n=8)
    env.reset(2000)
    assert env.channel_gain.shape == (8, 2) and np.all(env.channel_gain == 1e-5)
    env.step([SensorAction(0, 0, 0.0, 0.0)] * 8)
    assert np.all(env.channel_gain == 1e-5)
    base = _env(None, n=8)
    base.reset(2000)
    assert np.all(base.channel_gain < 1e-8)  # v1.0 distance law is untouched


def test_a1_const_outcome_sinr():
    out, _, _ = _first_step("A1-const", LONE)
    assert out[0].sinr == pytest.approx(1e-5 * 1.0 / NOISE)


def test_a3_total_bandwidth_is_62_5_mhz():
    v = variants()["A3-total"]
    assert v.subcarrier_bw(load_config(ROOT / "configs" / "lin2025.yaml")) == 62.5e6
    base, _, _ = _first_step(None, LONE)
    out, _, _ = _first_step("A3-total", LONE)
    assert out[0].sinr == base[0].sinr
    assert out[0].rate_bps == pytest.approx(62.5e6 * math.log2(1 + out[0].sinr))
    assert base[0].rate_bps == pytest.approx(250e6 * math.log2(1 + base[0].sinr))


@pytest.mark.parametrize(("name", "noise"), [("A2-dBm", 1e-13), ("A2-psd", 1e-12), ("V0", 1e-10)])
def test_a2_noise_values(name, noise):
    out, gains, _ = _first_step(name, LONE)
    assert out[0].sinr == pytest.approx(gains[0, 0] * 1.0 / noise)


def test_uav_cb_latency_single_and_shared():
    base, _, bits = _first_step(None, LONE)
    out, _, _ = _first_step("UAV-cb", LONE)
    extra = bits[0] * 24.5 / 1.4e9  # n_k = 1
    assert out[0].latency_s == pytest.approx(base[0].latency_s + extra)
    assert out[0].energy_j == base[0].energy_j  # energy unchanged
    assert out[0].rate_bps == base[0].rate_bps
    # two sensors on UAV 0 (different subcarriers) share the CPU: n_k = 2
    two = [SensorAction(1, 0, 1.0, 1.0), SensorAction(1, 1, 1.0, 1.0)]
    base2, _, bits2 = _first_step(None, two)
    out2, _, _ = _first_step("UAV-cb", two)
    for i in range(2):
        want = bits2[i] * 24.5 / (1.4e9 / 2)
        assert out2[i].latency_s == pytest.approx(base2[i].latency_s + want)
        assert out2[i].energy_j == base2[i].energy_j
    # one sensor per UAV: n_k = 1 for both
    sep = [SensorAction(1, 0, 1.0, 1.0), SensorAction(2, 0, 1.0, 1.0)]
    base3, _, bits3 = _first_step(None, sep)
    out3, _, _ = _first_step("UAV-cb", sep)
    for i in range(2):
        assert out3[i].latency_s == pytest.approx(base3[i].latency_s + bits3[i] * 24.5 / 1.4e9)


def test_n1_prop_env_matches_noma_group_with_rescaled_bandwidth():
    acts = [SensorAction(1, 0, 1.0, 0.8), SensorAction(1, 0, 0.5, 0.6)]
    out, gains, _ = _first_step("N1-prop", acts)
    g = np.array([gains[0, 0] * 1.0, gains[1, 0] * 0.5])
    bw = B * np.array([0.8, 0.6]) / 1.4
    sinr, rate, _ = noma_group(g, bw, NOISE, 10e6, sic=True)
    assert np.allclose([o.sinr for o in out], sinr)
    assert np.allclose([o.rate_bps for o in out], rate)


def test_n1_eq_env_ignores_bw_frac():
    acts = [SensorAction(1, 0, 1.0, 0.1), SensorAction(1, 0, 1.0, 0.9)]
    out, gains, _ = _first_step("N1-eq", acts)
    g = np.array([gains[0, 0], gains[1, 0]])
    _, rate, _ = noma_group(g, np.full(2, B / 2), NOISE, 10e6, sic=True)
    assert np.allclose([o.rate_bps for o in out], rate)


def test_a8lit_env_uses_own_gain_and_power_order():
    acts = [SensorAction(1, 0, 1.0, 1.0), SensorAction(1, 0, 0.5, 1.0)]
    out, gains, _ = _first_step("A8-lit", acts)
    sinr, _, _ = noma_group_a8lit(
        gains[:2, 0], np.array([1.0, 0.5]), np.full(2, B), NOISE, 10e6, sic=True
    )
    assert np.allclose([o.sinr for o in out], sinr)


# ------------------------------------------- controls and policy pairing
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("n", [2, 8, 12])
def test_local_computing_identical_under_every_variant(name, n):
    ref = episodes(variants()["V0"], "LocalComputing", n)
    got = episodes(variants()[name], "LocalComputing", n)
    for r, g in zip(ref, got, strict=True):
        for c in COLS:
            assert same(r[c], g[c]), (name, n, c)


def _record_actions(variant, seed, n=8):
    cfg = load_config(ROOT / "configs" / "lin2025.yaml", n_sensors=n)
    env = LinMECEnv(cfg, sic=True, mode="POS", s3_variant=variant)
    pol = POLICIES["RandomPolicy"]()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rec = []
    while True:
        acts = pol.act(env, obs, rng)
        rec.append(acts)
        obs, _, term, trunc, _ = env.step(acts)
        if term or trunc:
            return rec, rng.uniform()  # the next draw also shows the rng stream position


@pytest.mark.parametrize("name", ALL[1:])
def test_random_policy_actions_identical_between_v0_and_variant(name):
    for seed in (2000, 2001):
        a, tail_a = _record_actions(variants()["V0"], seed)
        b, tail_b = _record_actions(variants()[name], seed)
        assert a == b and tail_a == tail_b
