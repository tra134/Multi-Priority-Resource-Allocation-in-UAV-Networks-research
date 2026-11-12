"""train.py writes a provenance manifest next to every checkpoint (P-012)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def train_mod():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = {"meta": {"env_version": "env-v2", "env_frozen": True}, "sac": {"N_critics": 2}}


def _write(mod, tmp_path, step, **kw):
    ckpt = tmp_path / f"run_step{step}.pt"
    ckpt.write_bytes(b"x")
    args = dict(root=ROOT, cfg=CFG, run_name="run", seed=3, n_critics=2,
                mask_aware=False, dr_wide=True, step=step, total_steps=100,
                resumed_from_step=None)
    args.update(kw)
    mod.write_checkpoint_manifest([ckpt], **args)
    return ckpt


def test_manifest_written_next_to_checkpoint_and_history(train_mod, tmp_path):
    ckpt = _write(train_mod, tmp_path, 50)
    m = json.loads((tmp_path / "run_step50.manifest.json").read_text())
    assert m["run_name"] == "run" and m["seed"] == 3 and m["n_critics"] == 2
    assert m["step"] == 50 and m["total_steps"] == 100
    assert m["dr_wide"] is True and m["mask_aware"] is False
    assert m["env_version"] == "env-v2" and m["env_frozen"] is True
    assert m["resumed"] is False and m["resumed_from_step"] is None
    for k in ("git_commit", "git_dirty", "config_sha256", "timestamp_utc", "torch_version",
              "cuda_available", "cuda_device", "python", "replay_buffer_restored"):
        assert k in m
    assert m["checkpoint"] == ckpt.name
    _write(train_mod, tmp_path, 60)
    lines = (tmp_path / "run.manifest.jsonl").read_text().splitlines()
    assert [json.loads(x)["step"] for x in lines] == [50, 60]


def test_resumed_run_records_empty_replay_buffer(train_mod, tmp_path):
    _write(train_mod, tmp_path, 70, resumed_from_step=60)
    m = json.loads((tmp_path / "run_step70.manifest.json").read_text())
    assert m["resumed"] is True and m["resumed_from_step"] == 60
    assert m["replay_buffer_restored"] is False


def test_manifest_failure_warns_and_does_not_crash(train_mod, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("no git")
    monkeypatch.setattr(train_mod, "collect_manifest", boom)
    _write(train_mod, tmp_path, 80)
    assert "WARNING" in capsys.readouterr().out
    assert not (tmp_path / "run_step80.manifest.json").exists()


def test_short_real_run_writes_manifest(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg.setdefault("timing", {})["episode_slots"] = 20
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
         "--steps", "5", "--n-critics", "2", "--ckpt-every", "5", "--eval-every", "100000",
         "--eval-episodes", "1", "--allow-unfrozen",
         "--ckpt-dir", str(tmp_path / "ckpt"), "--tb-dir", str(tmp_path / "tb")],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    files = sorted(p.name for p in (tmp_path / "ckpt").iterdir())
    assert any(f.endswith("_step5.manifest.json") for f in files)
    assert any(f.endswith("_latest.manifest.json") for f in files)
    assert any(f.endswith(".manifest.jsonl") for f in files)
