"""train.py must refuse to train while the environment is not frozen."""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(config, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(config),
         "--steps", "1", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )


def test_refuses_when_environment_is_not_frozen(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["meta"] = {"env_version": "pending", "env_frozen": False}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = _run(path, "--ckpt-dir", str(tmp_path / "ckpt"), "--tb-dir", str(tmp_path / "tb"))
    assert out.returncode != 0
    assert "env_frozen" in (out.stderr + out.stdout)
    assert not (tmp_path / "ckpt").exists()
