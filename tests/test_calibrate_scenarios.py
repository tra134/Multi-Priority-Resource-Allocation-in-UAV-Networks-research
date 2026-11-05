"""Stage 2 final check: comparator N and the revised criteria (TE-015, TE-016)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cs", ROOT / "scripts" / "calibrate_scenarios.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def _shares(action):
    e = np.exp(3.0 * np.asarray(action, dtype=float))
    return e / e.sum()


def test_nominal_sized_policy_has_requested_emergency_share_and_ratio():
    pol = cs.build_policy("nominal_sized:0.1", CFG)
    a = pol(None, {})
    for part in (a[:3], a[3:]):          # bandwidth and power use the same shares
        s = _shares(part)
        assert abs(s[0] - 0.10) < 1e-6
        assert abs(s[1] / s[2] - 35.0 / 28.0) < 1e-5
        assert abs(s.sum() - 1.0) < 1e-9


def _frame(m1, nominal, fair, load_rate=117.9):
    rows = []

    def add(policy, scenario, viol_ood=np.nan, viol_ep=0.0, imp=30.0, norm=20.0,
            imp_ood=None, norm_ood=None, rate=load_rate):
        rows.append({"policy": policy, "scenario": scenario, "viol_ood": viol_ood,
                     "viol_episode": viol_ep, "delivered_imp_mbps": imp,
                     "delivered_norm_mbps": norm,
                     "delivered_imp_ood_mbps": imp if imp_ood is None else imp_ood,
                     "delivered_norm_ood_mbps": norm if norm_ood is None else norm_ood,
                     "allocated_rate_mbps": rate})

    add("fair_share", "S0")
    add("m1_queue_priority", "S0", viol_ep=0.004)
    for s in ("S1", "S2", "S4"):
        add("m1_queue_priority", s, viol_ood=m1[s], imp_ood=10.0, norm_ood=5.0)
        add("nominal", s, viol_ood=nominal[s])
        add("fair_share", s, viol_ood=fair[s])
    add("emergency_only", "S5", viol_ood=0.93)
    return pd.DataFrame(rows)


def test_criteria_pass_for_the_recorded_result():
    df = _frame({"S1": .009, "S2": .024, "S4": .158}, {"S1": .93, "S2": .82, "S4": .99},
                {"S1": .014, "S2": .083, "S4": .86})
    crit, load = cs.evaluate_final(df, CFG)
    assert all(ok for _, ok in crit.values()), crit
    assert 0.5 <= load <= 0.6


def test_c3_revised_fails_when_the_nominal_policy_is_not_clearly_worse():
    df = _frame({"S1": .009, "S2": .024, "S4": .158}, {"S1": .02, "S2": .82, "S4": .99},
                {"S1": .014, "S2": .083, "S4": .86})
    crit, _ = cs.evaluate_final(df, CFG)
    assert crit["C3_revised"][1] is False


def test_fair_share_no_longer_gates_c3():
    df = _frame({"S1": .009, "S2": .024, "S4": .158}, {"S1": .93, "S2": .82, "S4": .99},
                {"S1": .009, "S2": .024, "S4": .158})    # fair share as good as M1
    crit, _ = cs.evaluate_final(df, CFG)
    assert crit["C3_revised"][1] is True


def test_mismatch_with_recorded_values_is_detected():
    df = _frame({"S1": .009, "S2": .024, "S4": .158}, {"S1": .93, "S2": .82, "S4": .99},
                {"S1": .014, "S2": .083, "S4": .86})
    assert cs.compare_with_expected(df, 0.12)          # wrong r
    assert cs.compare_with_expected(df, 0.10)          # the synthetic frame is not the recorded one
