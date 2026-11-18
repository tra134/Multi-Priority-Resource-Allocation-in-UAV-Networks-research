import importlib.util
from pathlib import Path

import numpy as np
from scipy import stats

_p = Path(__file__).resolve().parents[1] / "scripts" / "s4_equivalence.py"
_s = importlib.util.spec_from_file_location("s4_equivalence", _p)
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)


def test_welch_tost_matches_scipy():
    a = [99.0, 99.5, 98.8, 99.2, 99.1]
    b = [98.9, 99.0, 99.3, 98.7, 99.4, 99.0]
    r = m.welch_tost(a, b, margin=1.0)
    tt = stats.ttest_ind(a, b, equal_var=False)
    assert np.isclose(r["diff"] / r["se"], tt.statistic)
    assert np.isclose(r["df"], tt.df)
    ci = tt.confidence_interval(0.90)
    assert np.isclose(r["ci90_lo"], ci.low) and np.isclose(r["ci90_hi"], ci.high)
    p1 = stats.ttest_ind(np.array(a) + 1.0, b, equal_var=False, alternative="greater").pvalue
    p2 = stats.ttest_ind(np.array(a) - 1.0, b, equal_var=False, alternative="less").pvalue
    assert np.isclose(r["p_tost"], max(p1, p2))
    assert r["equivalent"] == (r["ci90_lo"] > -1 and r["ci90_hi"] < 1)
    assert np.isclose(r["smallest_margin"], max(abs(r["ci90_lo"]), abs(r["ci90_hi"])))


def test_not_equivalent_when_far():
    r = m.welch_tost([90, 91, 92, 90, 91], [99, 99, 98, 99, 99])
    assert not r["equivalent"] and r["p_tost"] > 0.05
