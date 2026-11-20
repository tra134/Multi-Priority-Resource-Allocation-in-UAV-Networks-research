"""Tests of the T-5 replay (docs/lin2025_spec.md Section 13): Table 4 slot in LinMECEnv."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("run_t5", ROOT / "scripts" / "run_t5.py")
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)

T2 = pd.read_csv(ROOT / "results" / "T2_offload_physics_2026-11-13" / "table4_replay.csv")
T2 = T2[T2["cand"] == "a_dist_percarrier"]


@pytest.mark.parametrize("reading", ["R1", "R2"])
def test_replay_at_zero_reproduces_t2(reading):
    rows = m.replay(reading, loss_db=0.0)
    ref = T2[T2["reading"] == reading].set_index("sensor")
    assert [r["sensor"] for r in rows] == list(ref.index)
    for r in rows:
        e = ref.loc[r["sensor"]]
        assert r["sim_success"] == bool(e["sim_success"])
        assert r["match"] == bool(e["match"])
        assert r["mode"] == e["mode"]
        if r["mode"] == "offload":
            assert r["sinr"] == pytest.approx(e["sinr"], abs=2e-3)
            assert r["rate_mbps"] == pytest.approx(e["rate_mbps"], abs=0.02)


def test_t2_score_is_14_of_16_at_zero():
    assert m.score(m.replay("R1", 0.0)) == 7
    assert m.score(m.replay("R2", 0.0)) == 7


def test_loss_scales_gain_by_db():
    g0 = m.table4_gain(0.0)
    g6 = m.table4_gain(6.0)
    assert np.allclose(g6, g0 * 10 ** (-0.6))
    assert np.allclose(m.table4_gain(0.0), m.table4_gain(0.0, alpha=2.0))


def test_feasible_set_and_decision():
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    s1 = {0.5: 8, 0.75: 8, 1.0: 8, 1.5: 8}
    s2 = {0.5: 8, 0.75: 8, 1.0: 8, 1.25: 8, 1.5: 8}
    f = m.feasible(grid, s1, s2)
    assert f == [0.5, 0.75, 1.0, 1.5]
    d = m.decide(f, grid)
    assert d["adopted"] and d["interval"] == [0.5, 1.0]
    assert d["L_star"] == 1.0  # midpoint 0.75 -> nearest 0.5 dB, ties go up
    assert m.decide([0.0, 0.25], grid)["adopted"] is False  # 0 in F
    assert m.decide([], grid)["adopted"] is False  # empty


def test_rounding_midpoint_to_half_db():
    grid = [x * 0.25 for x in range(-40, 81)]
    f = [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5]  # midpoint 5.75 -> 6.0 (round to nearest 0.5, ties up)
    assert m.decide(f, grid)["L_star"] == 6.0
    f = [5.0, 5.25, 5.5]  # midpoint 5.25 -> 5.5 (5.25 is a tie between 5.0 and 5.5, ties up)
    assert m.decide(f, grid)["L_star"] == 5.5
    f = [6.0, 6.25, 6.5, 7.0, 7.25, 7.5, 7.75, 8.0]  # largest contiguous = 7.0..8.0, midpoint 7.5
    assert m.decide(f, grid)["L_star"] == 7.5
