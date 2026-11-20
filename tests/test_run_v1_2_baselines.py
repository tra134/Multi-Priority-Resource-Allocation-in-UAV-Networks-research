import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("run_v1_2_baselines", ROOT / "scripts" / "run_v1_2_baselines.py")
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)


def test_refuses_test_seeds_and_existing_dir(tmp_path):
    assert m.main(["--out", str(tmp_path / "a"), "--seeds-start", "5000", "--n-seeds", "2"]) == 2
    assert m.main(["--out", str(tmp_path)]) == 2


def test_tiny_run_local_identical_and_nosic_row(tmp_path):
    out = tmp_path / "o"
    assert m.main(["--out", str(out), "--n-seeds", "2", "--n-list", "8"]) == 0
    s = pd.read_csv(out / "summary.csv")
    loc = s[s["policy"] == "LocalComputing"].iloc[0]
    assert loc["success_rate_diff_mean"] == 0.0  # LocalComputing identical under v1.0 and v1.2
    assert (s["policy"] == m.NOSIC_LABEL).sum() == 1
    near = s[s["policy"] == "NearestUAVFullPower"].iloc[0]
    assert near["success_rate_v1_2_mean"] <= near["success_rate_v1_0_mean"]  # weaker links never help here
    man = json.loads((out / "manifest.json").read_text())
    assert man["llm_client"] == "none" and "git_dirty" in man
