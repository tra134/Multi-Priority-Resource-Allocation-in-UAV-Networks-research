"""calib-v2: per-checkpoint trigger and shield values (TE-071, TE-073)."""

from __future__ import annotations

import pytest
import yaml

from uavrl.calibration import (
    load_calibration,
    required_params,
    resolve_params,
)

NO_OVERRIDE = {"tau_unc": None, "theta_queue": None,
               "random_rate": None, "periodic_k": None}


def _write(tmp_path, ckpts):
    p = tmp_path / "calib.yaml"
    p.write_text(yaml.safe_dump({"version": "calib-v2", "checkpoints": ckpts}))
    return p


FULL = {"tau_unc": 0.3, "theta_queue": 20000.0, "m8_random_rate": 0.006, "m9_K": 150}
PENDING = {"tau_unc": 0.2, "theta_queue": 35000.0, "m8_random_rate": None,
           "m9_K": None, "status": "pending"}


def test_loader_returns_per_run_values(tmp_path):
    p = _write(tmp_path, {"sac_seed0_N10_mask": FULL, "sac_seed1_N10_mask": PENDING})
    calib = load_calibration(p)
    vals, src = resolve_params(calib, "sac_seed1_N10_mask", NO_OVERRIDE,
                               ("tau_unc", "theta_queue"))
    assert vals == {"tau_unc": 0.2, "theta_queue": 35000.0}
    assert all(s == "calibration" for s in src.values())
    vals0, _ = resolve_params(calib, "sac_seed0_N10_mask", NO_OVERRIDE,
                              ("random_rate", "periodic_k"))
    assert vals0 == {"random_rate": 0.006, "periodic_k": 150}


def test_override_beats_file(tmp_path):
    calib = load_calibration(_write(tmp_path, {"sac_seed0_N10_mask": FULL}))
    ov = dict(NO_OVERRIDE, tau_unc=0.9)
    vals, src = resolve_params(calib, "sac_seed0_N10_mask", ov,
                               ("tau_unc", "theta_queue"))
    assert vals["tau_unc"] == 0.9 and vals["theta_queue"] == 20000.0
    assert src == {"tau_unc": "flag", "theta_queue": "calibration"}


def test_override_fills_null(tmp_path):
    calib = load_calibration(_write(tmp_path, {"sac_seed1_N10_mask": PENDING}))
    ov = dict(NO_OVERRIDE, random_rate=0.01)
    vals, _ = resolve_params(calib, "sac_seed1_N10_mask", ov, ("random_rate",))
    assert vals == {"random_rate": 0.01}


def test_null_value_raises_clear_error(tmp_path):
    calib = load_calibration(_write(tmp_path, {"sac_seed1_N10_mask": PENDING}))
    with pytest.raises(SystemExit, match="random_rate.*sac_seed1_N10_mask"):
        resolve_params(calib, "sac_seed1_N10_mask", NO_OVERRIDE, ("random_rate",))


def test_missing_checkpoint_raises(tmp_path):
    calib = load_calibration(_write(tmp_path, {"sac_seed0_N10_mask": FULL}))
    with pytest.raises(SystemExit, match="sac_seed3_N10_mask"):
        resolve_params(calib, "sac_seed3_N10_mask", NO_OVERRIDE, ("tau_unc",))


def test_nothing_required_needs_no_file_entry():
    # static methods: no parameters needed, no entry, no error
    assert resolve_params({}, "sac_seed0_N10_mask", NO_OVERRIDE, ()) == ({}, {})


def test_required_params_by_method_kind():
    assert required_params({"kind": "static"}) == ()
    assert required_params({"kind": "agent"}) == ()
    assert required_params({"kind": "reserve"}) == ()
    assert required_params({"kind": "rule"}) == ("theta_queue",)
    assert set(required_params({"kind": "llm", "trigger": "hybrid"})) == {
        "tau_unc", "theta_queue"}
    assert required_params({"kind": "llm", "trigger": "random"}) == ("random_rate",)
    assert required_params({"kind": "llm", "trigger": "periodic"}) == ("periodic_k",)
    assert required_params({"kind": "llm", "trigger": "reactive"}) == ()
    assert required_params({"kind": "llm", "trigger": "always"}) == ()


def test_missing_file_is_empty_until_needed(tmp_path):
    assert load_calibration(tmp_path / "nope.yaml") == {}


def test_repo_yaml_parses_with_seeds_0_to_4():
    from uavrl.config import find_project_root
    path = find_project_root() / "configs" / "calibration_v2.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw["version"] == "calib-v2" and raw["env_version"] == "env-v2"
    calib = load_calibration(path)
    for s in range(5):
        e = calib[f"sac_seed{s}_N10_mask"]
        assert e["tau_unc"] > 0 and e["theta_queue"] in (20000, 35000)
    assert calib["sac_seed0_N10_mask"]["m9_K"] == 157
    assert calib["sac_seed1_N10_mask"]["m9_K"] is None
    assert "sources" in raw
