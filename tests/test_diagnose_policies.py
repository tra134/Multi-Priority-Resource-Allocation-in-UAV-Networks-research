"""Unit tests for the metric functions of scripts/diagnose_policies.py (S-5). Hand-built action lists only;
no test seeds (5000+) and no trained runs are used."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from uavmec.env.types import SensorAction

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("diagnose_policies", ROOT / "scripts" / "diagnose_policies.py")
dp = importlib.util.module_from_spec(_spec)
sys.modules["diagnose_policies"] = dp
_spec.loader.exec_module(dp)


def A(t, m=0, p=1.0, b=1.0):
    return SensorAction(t, m, p, b)


def test_seed_guard():
    dp.check_seeds(2000, 100)  # ok
    dp.check_seeds(4900, 100)  # ends at 4999, ok
    for start, n in [(5000, 1), (4950, 100), (6000, 10)]:
        with pytest.raises(ValueError):
            dp.check_seeds(start, n)
    with pytest.raises(ValueError):
        dp.check_seeds(2000, 0)


def test_main_refuses_test_seeds(tmp_path, capsys):
    assert dp.main(["--runs", str(tmp_path), "--out", str(tmp_path / "o"), "--seeds-start", "5000"]) == 2
    assert "refused" in capsys.readouterr().err


def test_deciles():
    d = dp.deciles(np.linspace(0, 1, 11))  # 0, .1, ..., 1
    assert d == pytest.approx([0.1 * i for i in range(1, 10)])
    assert all(math.isnan(x) for x in dp.deciles([]))


def test_shared_group_fraction():
    # slot 1: sensors 0,1 share (UAV 1, sub 0); sensor 2 alone on (UAV 1, sub 1); sensor 3 local.
    # slot 2: 3 sensors on different groups, none shared (same subcarrier but different UAVs).
    s1 = [A(1, 0), A(1, 0), A(1, 1), A(0)]
    s2 = [A(1, 0), A(2, 0), A(2, 1)]
    assert dp.shared_group_fraction([s1, s2]) == pytest.approx(2 / 6)
    assert dp.shared_group_fraction([[A(1, 0), A(1, 0), A(1, 0)]]) == 1.0  # a group of 3: all three count
    assert math.isnan(dp.shared_group_fraction([[A(0), A(0)]]))


def test_action_metrics_hand_case():
    # 2 slots x 3 sensors = 6 sensor-slots. nearest UAV (0-based): slot 1 [0, 1, 1]; slot 2 [0, 0, 1].
    nearest = [np.array([0, 1, 1]), np.array([0, 0, 1])]
    heur = [[A(1, 0), A(2, 0), A(2, 1)], [A(1, 0), A(1, 1), A(2, 0)]]  # always nearest, full power
    learner = [
        [A(1, 0, 0.5, 1.0), A(1, 0, 1.0, 0.5), A(0)],  # hit, wrong UAV (but offloaded), local
        [A(1, 2, 1.0, 1.0), A(1, 2, 0.0, 0.0), A(2, 1, 0.5, 0.5)],  # hit, hit, hit
    ]
    m = dp.action_metrics(learner, heur, nearest)
    assert m["n_sensor_slots"] == 6 and m["n_offloaded"] == 5
    assert m["offload_frac"] == pytest.approx(5 / 6)
    assert m["nearest_frac"] == pytest.approx(4 / 6)  # hits: s1 sensor0; s2 all three
    assert m["nearest_frac_of_offloaded"] == pytest.approx(4 / 5)
    assert m["agreement"] == pytest.approx(4 / 6)  # heuristic targets equal nearest+1 here
    assert m["power_mean"] == pytest.approx(np.mean([0.5, 1.0, 1.0, 0.0, 0.5]))
    assert m["bw_mean"] == pytest.approx(np.mean([1.0, 0.5, 1.0, 0.0, 0.5]))
    # shared groups: slot 1 (UAV1, sub0) x2 -> 2 shared; slot 2 (UAV1, sub2) x2 -> 2 shared; of 5 offloaded
    assert m["shared_group_frac"] == pytest.approx(4 / 5)
    assert m["heur_shared_group_frac"] == 0.0
    assert m["power_d50"] == pytest.approx(0.5)  # sorted [0, .5, .5, 1, 1] -> median 0.5
    assert m["bw_d10"] == pytest.approx(np.percentile([1.0, 0.5, 1.0, 0.0, 0.5], 10))


def test_action_metrics_all_local():
    nearest = [np.array([0, 1])]
    m = dp.action_metrics([[A(0, 0, 0.0, 0.0)] * 2], [[A(1), A(2)]], nearest)
    assert m["offload_frac"] == 0.0 and m["nearest_frac"] == 0.0 and m["agreement"] == 0.0
    assert math.isnan(m["power_mean"]) and math.isnan(m["bw_mean"]) and math.isnan(m["power_d50"])
    assert dp.reading_label(m) == "deviates from the heuristic"


def test_reading_label_thresholds():
    base = {"agreement": 0.9, "power_mean": 0.9, "bw_mean": 0.9}
    assert dp.reading_label(base) == "close to the heuristic"  # all three are inclusive (>=)
    for k in base:
        assert dp.reading_label({**base, k: 0.8999}) == "deviates from the heuristic"


def test_paired_difference():
    r = dp.paired_difference([0.9, 0.8, 1.0, 0.7], [0.8, 0.8, 0.9, 0.6], n_boot=2000)
    assert r["n_pairs"] == 4 and r["mean"] == pytest.approx(0.075)
    assert r["ci_lo"] <= r["mean"] <= r["ci_hi"]
    # constant difference: degenerate CI equal to it
    c = dp.paired_difference([0.5, 0.6, 0.7], [0.4, 0.5, 0.6], n_boot=500)
    assert c["mean"] == pytest.approx(0.1) and c["ci_lo"] == pytest.approx(0.1) and c["ci_hi"] == pytest.approx(0.1)
    # NaN pairs dropped; reproducible with the fixed RNG seed
    n = dp.paired_difference([0.5, float("nan"), 0.7], [0.4, 0.5, float("nan")], n_boot=100)
    assert n["n_pairs"] == 1
    assert dp.paired_difference([1, 2, 3], [0, 0, 0], n_boot=300) == dp.paired_difference([1, 2, 3], [0, 0, 0], n_boot=300)
    assert dp.paired_difference([float("nan")], [1.0])["n_pairs"] == 0
    assert dp.N_BOOT == 10_000 and dp.BOOT_SEED == 0
