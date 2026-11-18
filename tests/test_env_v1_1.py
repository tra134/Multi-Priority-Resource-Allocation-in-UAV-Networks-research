"""Acceptance tests for lin-env-v1.1 (docs/lin2025_spec.md Section 12, checks 1-5; A-42).

v1.1 = v1.0 + A-42 (proportional bandwidth rescaling on a shared carrier), selected only by
configs/lin2025_v1_1.yaml (LinConfig.bandwidth_sharing = "proportional").
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from uavmec.baselines.policies import POLICIES
from uavmec.config import LinConfig, load_config
from uavmec.env.lin_env import LinMECEnv, proportional_bandwidth
from uavmec.env.noma import noma_links
from uavmec.env.s3_variants import load_s3_variants
from uavmec.env.types import SensorAction
from uavmec.eval.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "configs" / "lin2025.yaml"
V11 = ROOT / "configs" / "lin2025_v1_1.yaml"
REF = json.loads((ROOT / "tests" / "data" / "v1_reference_s3.json").read_text())
SEEDS = range(2000, 2005)
COLS = ("success_rate", "mean_latency_ms", "mean_energy_mw", "n_local", "n_offload",
        "fail_deadline", "fail_energy", "fail_rate", "fail_decode", "n_tasks", "n_success")
B = 250.0e6


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def episodes_v10(policy, n, s3_variant=None):
    env = LinMECEnv(load_config(V10, n_sensors=n), sic=True, mode="POS", s3_variant=s3_variant)
    return [run_episode(env, POLICIES[policy](), s) for s in SEEDS]


def episodes_v11(policy, n):
    env = LinMECEnv(load_config(V11, n_sensors=n), sic=True, mode="POS")
    return [run_episode(env, POLICIES[policy](), s) for s in SEEDS]


def assert_rows_equal(a, b, ctx):
    for x, y in zip(a, b, strict=True):
        assert x["seed"] == y["seed"]
        for c in COLS:
            assert same(x[c], y[c]), (ctx, x["seed"], c, x[c], y[c])


# ------------------------------------------------------------ config
def test_default_is_independent_and_value_validated():
    assert LinConfig().bandwidth_sharing == "independent"
    assert load_config(V10).bandwidth_sharing == "independent"
    assert load_config(V11).bandwidth_sharing == "proportional"
    with pytest.raises(ValueError):
        LinConfig(bandwidth_sharing="equal")


def test_v1_1_config_differs_from_v1_0_only_in_bandwidth_sharing():
    a, b = dataclasses.asdict(load_config(V10)), dataclasses.asdict(load_config(V11))
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"bandwidth_sharing"}


# ------------------------------------------------- check 1: v1.0 unchanged
@pytest.mark.parametrize("n", [2, 8, 12])
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_check1_v1_0_matches_frozen_reference(policy, n):
    assert_rows_equal(REF[f"{policy}|{n}"], episodes_v10(policy, n), (policy, n))


# ------------------------------------ check 2: v1.1 == S-3 N1-prop per episode
@pytest.mark.parametrize("n", [2, 8, 12])
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_check2_v1_1_equals_s3_n1_prop(policy, n):
    n1prop = load_s3_variants(ROOT / "configs" / "s3_variants.yaml")["N1-prop"]
    assert_rows_equal(episodes_v10(policy, n, n1prop), episodes_v11(policy, n), (policy, n))


# ------------------------------------------- check 3: LocalComputing identical
@pytest.mark.parametrize("n", [2, 8, 12])
def test_check3_local_computing_identical(n):
    assert_rows_equal(episodes_v10("LocalComputing", n), episodes_v11("LocalComputing", n), n)


# -------------------------------------------------- check 4: A-42 allocation
def test_check4_single_sensor_gets_f_times_b():
    out = proportional_bandwidth(np.array([0]), np.array([0]), np.array([0.37]), B)
    assert np.array_equal(out, np.array([0.37 * B]))


def test_check4_group_sum_at_most_one_is_unchanged():
    f = np.array([0.3, 0.4])
    out = proportional_bandwidth(np.zeros(2, int), np.zeros(2, int), f, B)
    assert np.array_equal(out, f * B)
    f = np.array([0.6, 0.4])  # sum exactly 1.0
    assert np.array_equal(proportional_bandwidth(np.zeros(2, int), np.zeros(2, int), f, B), f * B)


def test_check4_group_sum_above_one_is_rescaled_to_b():
    f = np.array([0.5, 0.6, 0.4])
    out = proportional_bandwidth(np.zeros(3, int), np.zeros(3, int), f, B)
    assert np.allclose(out, B * f / 1.5)
    assert math.isclose(out.sum(), B, rel_tol=1e-12)
    assert np.allclose(out / out.sum(), f / f.sum())  # ratios of requests preserved


def test_check4_groups_are_separate_carriers():
    # (uav0,sub0): f .9+.9 -> rescaled; (uav0,sub1): f .5 alone; (uav1,sub0): f 1.0 alone
    k = np.array([0, 0, 0, 1])
    m = np.array([0, 0, 1, 0])
    f = np.array([0.9, 0.9, 0.5, 1.0])
    out = proportional_bandwidth(k, m, f, B)
    assert np.allclose(out, B * np.array([0.5, 0.5, 0.5, 1.0]))


def test_check4_zero_fractions_do_not_divide_by_zero():
    out = proportional_bandwidth(np.zeros(2, int), np.zeros(2, int), np.zeros(2), B)
    assert np.array_equal(out, np.zeros(2))


def test_check4_order_independent():
    rng = np.random.default_rng(1)
    n = 9
    k, m, f = rng.integers(0, 2, n), rng.integers(0, 3, n), rng.uniform(0.0, 1.0, n)
    base = proportional_bandwidth(k, m, f, B)
    for _ in range(20):
        p = rng.permutation(n)
        got = proportional_bandwidth(k[p], m[p], f[p], B)
        assert np.allclose(got, base[p], rtol=1e-12, atol=0.0)  # sum order may differ at 1 ulp


def test_check4_env_step_uses_rescaled_bandwidth():
    """Two sensors, same (UAV, subcarrier), f=1 each: each gets B/2 in v1.1 and B in v1.0."""
    acts = [SensorAction(1, 0, 1.0, 1.0), SensorAction(1, 0, 1.0, 1.0)]
    rates = {}
    for name, path in (("v10", V10), ("v11", V11)):
        env = LinMECEnv(load_config(path, n_sensors=2, n_uavs=1), sic=True, mode="POS")
        env.reset(seed=2000)
        gain = env.channel_gain[:, 0].copy()
        _, _, _, _, info = env.step(acts)
        rates[name] = [o.rate_bps for o in info["outcomes"]]
        bw = np.array([B, B]) if name == "v10" else np.array([B / 2, B / 2])
        _, exp, _ = noma_links(np.zeros(2, int), np.zeros(2, int), gain * 1.0, bw, 1e-10, 10.0e6, True)
        assert np.allclose(rates[name], exp, rtol=1e-12)
    assert rates["v10"] != rates["v11"]


def test_check4_single_offloader_same_in_both_versions():
    acts = [SensorAction(1, 0, 1.0, 0.5), SensorAction(0, 0, 0.0, 0.0)]
    outs = []
    for path in (V10, V11):
        env = LinMECEnv(load_config(path, n_sensors=2, n_uavs=1), sic=True, mode="POS")
        env.reset(seed=2001)
        outs.append(env.step(acts)[4]["outcomes"][0].rate_bps)
    assert outs[0] == outs[1]


# ------------------------------------------- check 5: env builds from the v1.1 config
def test_check5_env_builds_and_steps_from_v1_1_config():
    cfg = load_config(V11)
    env = LinMECEnv(cfg, sic=True, mode="POS")
    obs, _ = env.reset(seed=2000)
    assert len(obs) == cfg.n_sensors and obs[0].shape == (env.obs_dim,)
    acts = [SensorAction(1 + n % cfg.n_uavs, n % cfg.n_subcarriers, 1.0, 1.0) for n in range(cfg.n_sensors)]
    _, rew, _, trunc, info = env.step(acts)
    assert rew.shape == (cfg.n_sensors,) and not trunc and len(info["outcomes"]) == cfg.n_sensors


def test_s3_variant_cannot_be_combined_with_a42_config():
    v0 = load_s3_variants(ROOT / "configs" / "s3_variants.yaml")["V0"]
    with pytest.raises(ValueError):
        LinMECEnv(load_config(V11), s3_variant=v0)


def test_v1_1_really_differs_from_v1_0_when_carriers_are_shared():
    a, b = episodes_v10("NearestUAVFullPower", 12), episodes_v11("NearestUAVFullPower", 12)
    assert any(x["mean_latency_ms"] != y["mean_latency_ms"] for x, y in zip(a, b, strict=True))


def test_run_heuristics_refuses_test_seeds_and_existing_folder(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_heuristics", ROOT / "scripts" / "run_heuristics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp_path / "new"), "--seeds-start", "4999", "--n-seeds", "2"]) == 2
    assert mod.main(["--out", str(tmp_path), "--n-seeds", "2"]) == 2  # exists
    assert not (tmp_path / "new").exists()
