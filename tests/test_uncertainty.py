"""Tests for uncertainty calibration (plan.md Phase 5)."""

from __future__ import annotations

import numpy as np

from uavrl.uncertainty.calibration import (
    collect_trajectory,
    normalize_U,
    percentile_U,
    ood_metrics,
    auroc,
)


def test_auroc_perfect_and_random():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0  # perfectly separable
    scores2 = np.array([0.1, 0.4, 0.35, 0.8])
    labels2 = np.array([0, 0, 1, 1])
    assert abs(auroc(scores2, labels2) - 0.75) < 1e-9


def test_auroc_all_one_class_nan():
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_ood_metrics_shapes():
    rows = [
        {"U": 0.1, "meanQ": 1.0, "ood_flag": False, "delay_emg_s": 0.005,
         "queue_emg_bits": 1.0, "reward": 0.0, "slot": 0},
        {"U": 0.9, "meanQ": 1.0, "ood_flag": True, "delay_emg_s": 0.05,
         "queue_emg_bits": 5.0, "reward": -1.0, "slot": 1},
    ]
    m = ood_metrics(rows, tau=0.5, normalized=False)
    assert bool(m["trigger"][0]) is False and bool(m["trigger"][1]) is True
    assert m["first_trigger"] == 1
    assert m["first_violation"] == 1
    assert m["lead_time"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_normalize_U():
    rows = [{"U": 2.0, "meanQ": 4.0}, {"U": 1.0, "meanQ": -1.0}]
    n = normalize_U(rows, c=1.0)
    assert abs(n[0] - 2.0 / 5.0) < 1e-9
    assert abs(n[1] - 1.0 / 2.0) < 1e-9


def test_percentile_U():
    trajs = [
        [{"U": 1.0, "meanQ": 0.0}, {"U": 2.0, "meanQ": 0.0}],
        [{"U": 3.0, "meanQ": 0.0}],
    ]
    for r in trajs[0]:
        r["U_norm"] = r["U"]
    for r in trajs[1]:
        r["U_norm"] = r["U"]
    assert abs(percentile_U(trajs, q=50.0) - 2.0) < 1e-9


def test_collect_trajectory_runs(monkeypatch):
    # Use a tiny fake agent with select_action/uncertainty and a fake env.
    class FakeAgent:
        def select_action(self, obs, deterministic=True):
            return np.zeros(6, dtype=np.float32)

        def uncertainty(self, obs, action):
            return 0.5, 1.0

    class FakeEnv:
        def __init__(self):
            self.slot = 0

        def reset(self):
            self.slot = 0
            return np.zeros(17, dtype=np.float32), {}

        def step(self, action):
            self.slot += 1
            info = {
                "slot": self.slot, "ood_flag": False,
                "delay_s": np.array([0.005, 0.0, 0.0]),
                "queue_bits": np.array([1.0, 0.0, 0.0]),
            }
            return np.zeros(17, dtype=np.float32), 0.0, self.slot >= 3, False, info

    rows = collect_trajectory(FakeEnv(), FakeAgent(), deterministic=True)
    assert len(rows) == 3
    assert all(r["U"] == 0.5 for r in rows)
