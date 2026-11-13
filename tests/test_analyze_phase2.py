"""scripts/analyze_phase2.py on synthetic frames (no real data)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ap2", ROOT / "scripts" / "analyze_phase2.py")
ap2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ap2)

ALL_METHODS = (list(ap2.LLM_METHODS) + list(ap2.NON_LLM_METHODS))
N_BOOT = 200


def _frame(methods=ALL_METHODS, client="real", seeds=3, episodes=3, thr_m12=9.0):
    rng = np.random.default_rng(0)
    rows = []
    for m in methods:
        is_llm = m in ap2.LLM_METHODS
        for sc in ("S1", "S2", "S4"):
            for s in range(seeds):
                for e in range(episodes):
                    thr = thr_m12 if m == "m12_proposed" else 10.0
                    rows.append({
                        "method": m, "scenario": sc, "seed": s, "episode": e,
                        "eval_seed": 1000 * s + e,
                        "client": client if is_llm else "none",
                        "sos_violation_ood": 0.2 + 0.05 * s + rng.normal(0, 0.02)
                        + (0.1 if m == "m3_pure_sac" else 0.0),
                        "delivered_imp_ood_mbps": thr / 2,
                        "delivered_norm_ood_mbps": thr / 2 + rng.normal(0, 0.01),
                        "llm_calls": 10.0 if is_llm else 0.0,
                        "llm_call_rate": 0.005,
                    })
    return pd.DataFrame(rows)


def test_mock_llm_rows_refused():
    with pytest.raises(ap2.AnalysisError, match="client == 'real'"):
        ap2.analyze(_frame(client="mock"), n_boot=N_BOOT)


def test_allow_mock_stamps_outputs():
    tables, _, _, mock = ap2.analyze(_frame(client="mock"), n_boot=N_BOOT, allow_mock=True)
    assert mock
    for name in ("confirmatory_holm", "throughput_loss_vs_m3", "llm_cost"):
        assert (tables[name]["mock_note"] == ap2.MOCK_NOTE).all()


def test_always_llm_method_refused():
    df = pd.concat([_frame(), _frame(methods=["m11_always_llm"])], ignore_index=True)
    with pytest.raises(ap2.AnalysisError, match="forbidden"):
        ap2.analyze(df, n_boot=N_BOOT)


def test_holm_family_size_is_12_and_complete():
    tables, miss, _, mock = ap2.analyze(_frame(), n_boot=N_BOOT)
    t = tables["confirmatory_holm"]
    assert not mock and miss == []
    assert len(t) == 12
    assert (t["family_size"] == 12).all()
    assert set(t["baseline"]) == set(ap2.MAIN_BASELINES)
    assert (t["note"] == "").all()


def test_missing_comparison_detected_and_family_stays_12():
    df = _frame()
    df = df[~((df["method"] == "m4_redq_sac") & (df["scenario"] == "S2"))]
    tables, miss, _, _ = ap2.analyze(df, n_boot=N_BOOT)
    assert miss == [{"scenario": "S2", "main": "m12_proposed", "baseline": "m4_redq_sac"}]
    t = tables["confirmatory_holm"]
    assert len(t) == 11
    assert (t["family_size"] == 12).all()
    # fixed family: adjusted p uses m = 12, so it is at least the unpadded Holm value
    unpadded, _ = ap2.holm_bonferroni(t["p_value"].to_numpy())
    assert (t["p_holm"].to_numpy() >= unpadded - 1e-12).all()


def test_main_exits_nonzero_on_missing_unless_allowed(tmp_path):
    df = _frame()
    df = df[df["method"] != "m13_fixed_reserve"]
    f = tmp_path / "m.csv"
    df.to_csv(f, index=False)
    out = tmp_path / "o"
    assert ap2.main(["--in", str(f), "--out", str(out), "--n-boot", "100"]) == 2
    assert not (out / "confirmatory_holm.csv").exists()
    assert ap2.main(["--in", str(f), "--out", str(out), "--n-boot", "100",
                     "--allow-missing"]) == 0
    assert len(pd.read_csv(out / "confirmatory_holm.csv")) == 9
    assert (out / "missing_comparisons.csv").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["inputs"][0]["sha256"] and man["n_boot"] == 100 and man["seed"] == 12345


def test_main_refuses_mock_with_exit_3(tmp_path):
    f = tmp_path / "m.csv"
    _frame(client="mock").to_csv(f, index=False)
    assert ap2.main(["--in", str(f), "--out", str(tmp_path / "o")]) == 3


def test_exploratory_rows_outside_holm_family():
    tables, _, miss_e, _ = ap2.analyze(_frame(), n_boot=N_BOOT)
    e = tables["exploratory"]
    assert miss_e == []
    assert (e["family"] == "exploratory").all()
    assert "p_holm" not in e.columns and "reject_holm" not in e.columns
    assert e["note"].str.contains("exploratory").all()
    assert "m7_rule_shield_fitted" in set(e["baseline"])
    c = tables["confirmatory_holm"]
    assert "m7_rule_shield_fitted" not in set(c["baseline"])
    assert not (set(c["main"]) - {"m12_proposed"})


def test_throughput_loss_sign():
    # M12 delivers 9.0 against M3 10.0: a 10 % loss, positive
    tables, _, _, _ = ap2.analyze(_frame(thr_m12=9.0), n_boot=N_BOOT)
    t = tables["throughput_loss_vs_m3"]
    assert len(t) == 3
    assert np.allclose(t["loss_pct"], 10.0, atol=0.5)
    # M12 delivers more than M3: negative loss
    tables, _, _, _ = ap2.analyze(_frame(thr_m12=11.0), n_boot=N_BOOT)
    assert (tables["throughput_loss_vs_m3"]["loss_pct"] < 0).all()
    assert tables["throughput_loss_vs_m3"]["throughput_target_met"].all()


def test_nan_counts_reported():
    df = _frame()
    df.loc[(df["method"] == "m12_proposed") & (df["scenario"] == "S1")
           & (df["seed"] == 0) & (df["episode"] == 0), "sos_violation_ood"] = np.nan
    tables, _, _, _ = ap2.analyze(df, n_boot=N_BOOT)
    n = tables["nan_counts"]
    hit = n[(n["method"] == "m12_proposed") & (n["scenario"] == "S1")
            & (n["column"] == "sos_violation_ood")]
    assert int(hit["n_nan"].iloc[0]) == 1
