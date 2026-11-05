"""Stage 3 pilot report: validation seeds only, gate logic, no overwriting."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from uavrl.agents.sac import SACAgent
from uavrl.config import checkpoint_run_name

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pr", ROOT / "scripts" / "pilot_report.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def _save_untrained(ckpt_dir, n):
    name = checkpoint_run_name("sac", 0, n)
    agent = SACAgent(obs_dim=17, act_dim=6, device="cpu", seed=0, N_critics=n,
                     rng=np.random.default_rng(0))
    run_dir = Path(ckpt_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(run_dir / f"{name}_latest.pt"), step=0)


def test_end_to_end_with_untrained_checkpoints(tmp_path):
    ckpt = tmp_path / "ckpt"
    _save_untrained(ckpt, 2)
    _save_untrained(ckpt, 10)
    out = tmp_path / "out"
    verdict = pr.run(["--ckpt-dir", str(ckpt), "--episodes", "1", "--slots", "1500",
                      "--out", str(out)])
    assert (out / "pilot_report.csv").exists() and (out / "manifest.jsonl").exists()
    saved = json.loads((out / "pilot_report.json").read_text())
    assert saved["seeds"] == "val_seed(500..500)"
    assert set(saved["gates"]) == {"sac_n2", "sac_n10"}
    assert isinstance(verdict["stage3_pass"], bool)
    df = pd.read_csv(out / "pilot_report.csv")
    assert set(df.policy) == {"sac_n2", "sac_n10", "fair_share", "m1", "nominal"}
    assert set(df.scenario) == {"S0", "S1", "S2", "S4"}
    # ensemble uncertainty only exists for the learned policies
    assert df[df.policy == "sac_n10"].mean_U.notna().all()
    assert df[df.policy == "fair_share"].mean_U.isna().all()


def test_refuses_to_overwrite_and_needs_a_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        pr.run(["--ckpt-dir", str(tmp_path / "none"), "--out", str(tmp_path / "o1")])
    out = tmp_path / "o2"
    out.mkdir()
    (out / "pilot_report.json").write_text("{}")
    with pytest.raises(SystemExit):
        pr.run(["--ckpt-dir", str(tmp_path), "--out", str(out)])


def _table(sac_s0=0.01, sac_s1=0.5, sac_s2=0.02, sac_reward=1.0, imp=30.0, norm=20.0):
    rows = []

    def add(policy, scenario, viol_ep, viol_ood, reward, imp=30.0, norm=20.0):
        rows.append({"policy": policy, "scenario": scenario,
                     "sos_violation_episode": viol_ep, "sos_violation_ood": viol_ood,
                     "mean_reward": reward, "delivered_imp_mbps": imp,
                     "delivered_norm_mbps": norm, "mean_b_share_emg": 0.1,
                     "mean_p_share_emg": 0.1, "mean_U": math.nan})

    add("fair_share", "S0", 0.002, math.nan, 0.5)
    add("m1", "S0", 0.004, math.nan, 0.6)
    add("sac_n2", "S0", sac_s0, math.nan, sac_reward, imp, norm)
    for s, v in (("S1", sac_s1), ("S2", sac_s2)):
        add("sac_n2", s, 0.1, v, 0.0)
    return pd.DataFrame(rows)


def test_gates_pass_when_sac_learns_and_degrades_under_surge():
    gates, ok = pr.compute_gates(_table(), CFG, ["sac_n2"], 0.10)
    g = gates["sac_n2"]
    assert g["G1_learning"]["pass"] and g["G2_service"]["pass"] and g["G3_nominal_safety"]["pass"]
    assert g["G4_degradation"]["S1"]["pass"] and not g["G4_degradation"]["S2"]["pass"]
    assert ok is True


@pytest.mark.parametrize("kwargs, gate", [
    ({"sac_reward": 0.4}, "G1_learning"),                  # worse than fair share
    ({"imp": 10.0}, "G2_service"),                         # serves < 50% of 35 Mbps
    ({"norm": 10.0}, "G2_service"),                        # serves < 50% of 28 Mbps
    ({"sac_s0": 0.08}, "G3_nominal_safety"),               # > 5% on nominal
])
def test_each_gate_can_fail_on_its_own(kwargs, gate):
    gates, ok = pr.compute_gates(_table(**kwargs), CFG, ["sac_n2"], 0.10)
    assert gates["sac_n2"][gate]["pass"] is False
    assert ok is False


def test_g4_needs_both_the_ratio_and_the_five_point_gap():
    # 3x ratio met but only 2 points higher
    g, _ = pr.compute_gates(_table(sac_s0=0.01, sac_s1=0.03), CFG, ["sac_n2"], 0.10)
    assert g["sac_n2"]["G4_degradation"]["S1"]["pass"] is False
    # 6 points higher but below the 3x ratio
    g, _ = pr.compute_gates(_table(sac_s0=0.04, sac_s1=0.10), CFG, ["sac_n2"], 0.10)
    assert g["sac_n2"]["G4_degradation"]["S1"]["pass"] is False
    # S2 alone is enough for the gate, S1 is reported separately
    g, ok = pr.compute_gates(_table(sac_s1=0.02, sac_s2=0.4), CFG, ["sac_n2"], 0.10)
    assert g["sac_n2"]["G4_degradation"]["pass"] is True
    assert g["sac_n2"]["G4_degradation"]["S1"]["pass"] is False
    assert "S2 but not S1" in pr.advice(g, ["sac_n2"])
