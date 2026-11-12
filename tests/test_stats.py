"""Stage 8 statistics: hierarchical bootstrap, paired comparison, Holm."""

import numpy as np
import pandas as pd

from uavrl.eval.stats import (
    comparison_table,
    hierarchical_bootstrap_ci,
    holm_bonferroni,
    paired_comparison,
)


def _frame(method, offset, seeds=5, episodes=4, scenario="S1", rng_seed=0,
           metric="sos_violation_ood"):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in range(seeds):
        for e in range(episodes):
            rows.append({"method": method, "scenario": scenario, "seed": s,
                         "episode": e, "eval_seed": 100 * s + e,
                         metric: offset + 0.1 * s + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_holm_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005]
    adj, rej = holm_bonferroni(p, alpha=0.05)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert list(rej) == [True, False, False, True]


def test_holm_monotone_and_capped():
    adj, _ = holm_bonferroni([0.5, 0.9, 0.001])
    assert adj.max() <= 1.0
    assert np.isclose(adj[2], 0.003)


def test_bootstrap_ci_contains_true_mean():
    df = _frame("a", 1.0)
    r = hierarchical_bootstrap_ci(df, "sos_violation_ood", n_boot=2000, seed=1)
    assert r["ci_low"] <= 1.2 <= r["ci_high"]
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    assert r["n_seeds"] == 5


def test_bootstrap_is_reproducible():
    df = _frame("a", 1.0)
    a = hierarchical_bootstrap_ci(df, "sos_violation_ood", n_boot=300, seed=7)
    b = hierarchical_bootstrap_ci(df, "sos_violation_ood", n_boot=300, seed=7)
    assert a == b


def test_unit_is_training_seed():
    # Between-seed spread dominates; the CI must be wide even with many episodes.
    many = _frame("a", 0.0, seeds=5, episodes=200)
    r = hierarchical_bootstrap_ci(many, "sos_violation_ood", n_boot=1000, seed=0)
    assert r["ci_high"] - r["ci_low"] > 0.1  # seed means range 0..0.4


def test_paired_sign_and_relative_improvement():
    b = _frame("m3", 1.0, rng_seed=1)  # larger violation
    a = _frame("m12", 0.5, rng_seed=2)
    r = paired_comparison(a, b, "sos_violation_ood", n_boot=1000, seed=0)
    assert r["diff"] < 0  # a - b
    assert r["rel_improvement"] > 0  # (V_b - V_a) / V_b
    assert r["rel_ci_low"] <= r["rel_improvement"] <= r["rel_ci_high"]
    assert 0 <= r["p_value"] <= 1


def test_paired_no_difference_gives_large_p():
    a = _frame("m12", 1.0, rng_seed=1)
    b = a.copy()
    b["method"] = "m3"
    r = paired_comparison(a, b, "sos_violation_ood", n_boot=500, seed=0)
    assert r["diff"] == 0
    assert r["p_value"] > 0.5


def test_paired_requires_matching_keys():
    a = _frame("m12", 1.0)
    b = _frame("m3", 1.0).iloc[:5]
    r = paired_comparison(a, b, "sos_violation_ood", n_boot=200, seed=0)
    assert r["n_pairs"] == 5


def test_comparison_table_is_12_rows_with_holm():
    frames = []
    for sc in ("S1", "S2", "S4"):
        for i, m in enumerate(("m12_proposed", "m3_pure_sac", "m4_redq_sac",
                               "m7_rule_shield", "m13_fixed_reserve")):
            frames.append(_frame(m, 0.5 + 0.2 * i, scenario=sc, rng_seed=i))
    df = pd.concat(frames)
    tab = comparison_table(df, n_boot=200, seed=0)
    assert len(tab) == 12
    assert {"p_holm", "reject_holm", "rel_improvement", "scenario",
            "baseline"} <= set(tab.columns)
    assert (tab["p_holm"] >= tab["p_value"] - 1e-12).all()


def test_comparison_table_skips_mock_rows():
    df = _frame("m12_proposed", 0.5)
    df["client"] = "mock"
    base = _frame("m3_pure_sac", 1.0)
    base["client"] = "none"
    tab = comparison_table(pd.concat([df, base]), n_boot=100, seed=0,
                           scenarios=("S1",), baselines=("m3_pure_sac",))
    assert tab["note"].str.contains("mock").all()
