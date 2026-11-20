"""Acceptance tests for lin-env-v1.2 (docs/lin2025_spec.md Section 13; A-44 extra link loss)."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from uavmec.baselines.policies import POLICIES
from uavmec.config import LinConfig, load_config
from uavmec.env.channel import channel_gain
from uavmec.env.lin_env import LinMECEnv
from uavmec.env.s3_variants import load_s3_variants
from uavmec.env.types import SensorAction
from uavmec.eval.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "configs" / "lin2025.yaml"
V12 = ROOT / "configs" / "lin2025_v1_2.yaml"
REF = json.loads((ROOT / "tests" / "data" / "v1_reference_s3.json").read_text())
SEEDS = range(2000, 2005)
COLS = ("success_rate", "mean_latency_ms", "mean_energy_mw", "n_local", "n_offload",
        "fail_deadline", "fail_energy", "fail_rate", "fail_decode", "n_tasks", "n_success")


def same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def episodes(cfg_path, policy, n, **kw):
    env = LinMECEnv(load_config(cfg_path, n_sensors=n), sic=True, mode="POS")
    return [run_episode(env, POLICIES[policy](), s) for s in SEEDS]


def assert_rows_equal(a, b, ctx):
    for x, y in zip(a, b, strict=True):
        for c in COLS:
            assert same(x[c], y[c]), (ctx, x["seed"], c, x[c], y[c])


# ------------------------------------------------------------ config
def test_default_loss_is_zero_and_validated():
    assert LinConfig().extra_link_loss_db == 0.0
    assert load_config(V10).extra_link_loss_db == 0.0
    with pytest.raises(ValueError):
        LinConfig(extra_link_loss_db=float("nan"))
    with pytest.raises(ValueError):
        LinConfig(extra_link_loss_db=float("inf"))


def test_v1_2_config_differs_from_v1_0_only_in_extra_link_loss():
    a, b = dataclasses.asdict(load_config(V10)), dataclasses.asdict(load_config(V12))
    assert {k for k in a if a[k] != b[k]} == {"extra_link_loss_db"}
    assert b["extra_link_loss_db"] == 7.0  # L* from T-5 (results/t5_table4_snr_2026-11-20)
    assert b["bandwidth_sharing"] == "independent"  # A-42 OFF


def test_guard_rejects_loss_with_s3_variant():
    cfg = load_config(V12)
    v = load_s3_variants(ROOT / "configs" / "s3_variants.yaml")["N1-prop"]
    with pytest.raises(ValueError):
        LinMECEnv(cfg, s3_variant=v)


# ------------------------------------------------ check 4: A-44 unit tests
def test_channel_gain_zero_loss_is_identity():
    s, u = np.array([[1.0, 2.0], [50.0, 60.0]]), np.array([[30.0, 40.0]])
    assert np.array_equal(channel_gain(s, u, 100.0, 1e-5, 0.0), channel_gain(s, u, 100.0, 1e-5))


@pytest.mark.parametrize("loss", [-3.0, 3.0, 7.0, 20.0])
def test_channel_gain_scales_by_db(loss):
    s, u = np.array([[1.0, 2.0], [50.0, 60.0]]), np.array([[30.0, 40.0], [5.0, 5.0]])
    g0 = channel_gain(s, u, 100.0, 1e-5)
    assert np.allclose(channel_gain(s, u, 100.0, 1e-5, loss), g0 * 10 ** (-loss / 10), rtol=1e-12)


def test_env_gain_and_step_use_loss():
    cfg0, cfg7 = load_config(V10), load_config(V12)
    e0, e7 = LinMECEnv(cfg0), LinMECEnv(cfg7)
    e0.reset(seed=3), e7.reset(seed=3)
    assert np.allclose(e7.channel_gain, e0.channel_gain * 10 ** (-0.7))
    acts = [SensorAction(1, n % 4, 1.0, 0.5) for n in range(cfg0.n_sensors)]
    o0 = e0.step(acts)[4]["outcomes"]
    o7 = e7.step(acts)[4]["outcomes"]
    # weaker links: SINR lower for every offloader (same positions, tasks and actions)
    assert all(b.sinr < a.sinr for a, b in zip(o0, o7, strict=True))


# ------------------------------------------- check 1 and 3: v1.0 frozen, Local identical
@pytest.mark.parametrize("n", [2, 8, 12])
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_check1_v1_0_matches_frozen_reference(policy, n):
    assert_rows_equal(REF[f"{policy}|{n}"], episodes(V10, policy, n), (policy, n))


@pytest.mark.parametrize("n", [2, 8, 12])
def test_check3_local_computing_identical(n):
    assert_rows_equal(episodes(V10, "LocalComputing", n), episodes(V12, "LocalComputing", n), n)
