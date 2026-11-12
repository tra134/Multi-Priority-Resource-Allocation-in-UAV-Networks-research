"""Smoke test for scripts/make_figures.py on a tiny synthetic CSV."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"


def _synthetic(path):
    rng = np.random.default_rng(0)
    rows = []
    for sc in ("S1", "S2", "S4"):
        for i, m in enumerate(("m12_proposed", "m3_pure_sac", "m4_redq_sac",
                               "m7_rule_shield", "m13_fixed_reserve")):
            for s in range(3):
                for e in range(2):
                    rows.append({
                        "method": m, "method_label": m, "scenario": sc,
                        "seed": s, "episode": e, "eval_seed": 10 * s + e,
                        "client": "mock" if m == "m12_proposed" else "none",
                        "sos_violation_ood": 0.1 * i + rng.normal(0, 0.01),
                        "delivered_imp_ood_mbps": 5 + rng.normal(0, 0.1),
                        "delivered_norm_ood_mbps": 3 + rng.normal(0, 0.1),
                    })
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)


def test_requires_out():
    r = subprocess.run([sys.executable, str(SCRIPT), "--in", "x"],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_smoke(tmp_path):
    _synthetic(tmp_path / "grid")
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--in", str(tmp_path / "grid"),
         "--out", str(out), "--n-boot", "100"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name in ("violation_bars.png", "violation_bars.pdf",
                 "throughput_vs_violation.png", "holm_table.csv",
                 "holm_table.md"):
        assert (out / name).exists(), name
    md = (out / "holm_table.md").read_text()
    assert "mock" in md
    assert len(pd.read_csv(out / "holm_table.csv")) == 12
