import math

import pytest

from uavmec.env.types import TaskOutcome
from uavmec.eval.metrics import aggregate_episodes, bootstrap_ci, summarize_outcomes


def mk(mode="local", ok=True, lat=0.01, en=0.001, reason=None):
    return TaskOutcome(1e6, 0.05, mode, None, None, None, None, None, lat, en, ok, reason)


def test_summarize_hand_computed():
    slots = [
        [mk(lat=0.010, en=0.002), mk("offload", False, 0.5, 0.004, "rate")],
        [mk(lat=0.020, en=0.000), mk("offload", False, 0.5, 0.002, "deadline")],
    ]
    s = summarize_outcomes(slots)
    assert s["n_tasks"] == 4 and s["n_success"] == 2
    assert s["success_rate"] == 0.5
    assert s["mean_latency_ms"] == pytest.approx(15.0)
    # A-34 (T-3): energy averaged over successful tasks only, like latency: (2+0)/2 mJ
    assert s["mean_energy_mw"] == pytest.approx(1.0)
    assert s["n_local"] == 2 and s["n_offload"] == 2
    assert s["fail_rate"] == 1 and s["fail_deadline"] == 1 and s["fail_decode"] == 0


def test_no_success_latency_nan():
    s = summarize_outcomes([[mk(ok=False, reason="energy")]])
    assert s["success_rate"] == 0.0 and math.isnan(s["mean_latency_ms"])


def test_aggregate_pools_counts():
    a = summarize_outcomes([[mk(lat=0.010), mk(ok=False, reason="rate")]])
    b = summarize_outcomes([[mk(lat=0.030), mk(lat=0.030), mk(lat=0.030)]])
    g = aggregate_episodes([a, b])
    assert g["n_tasks"] == 5 and g["success_rate"] == pytest.approx(0.8)
    assert g["mean_latency_ms"] == pytest.approx((10 + 90) / 4)


def test_aggregate_energy_weighted_by_successes():
    a = summarize_outcomes([[mk(en=0.002), mk(ok=False, en=0.050, reason="rate")]])
    b = summarize_outcomes([[mk(en=0.004), mk(en=0.004), mk(en=0.004)]])
    g = aggregate_episodes([a, b])
    assert g["mean_energy_mw"] == pytest.approx((2 + 12) / 4)  # A-34: failed task's 50 mJ excluded


def test_bootstrap_ci_fixed_and_contains_mean():
    v = [0.2, 0.4, 0.6, 0.8]
    assert bootstrap_ci(v) == bootstrap_ci(v)
    lo, hi = bootstrap_ci(v)
    assert lo <= 0.5 <= hi
    assert bootstrap_ci([0.3] * 5) == (0.3, 0.3)
