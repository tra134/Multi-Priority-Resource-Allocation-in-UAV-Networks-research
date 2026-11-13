import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("uavmec.env.lin_env")

from uavmec.env.lin_env import LinMECEnv

from uavmec.baselines.policies import RandomPolicy
from uavmec.config import load_config
from uavmec.eval.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]


def factory():
    return LinMECEnv(load_config(), sic=True, mode="POS")


def test_evaluate_deterministic():
    a = evaluate(factory, RandomPolicy(), [0, 1, 2])
    b = evaluate(factory, RandomPolicy(), [0, 1, 2])
    assert len(a) == 3
    assert a.equals(b)


@pytest.mark.parametrize("script", ["evaluate_baselines.py", "calibrate_local.py"])
def test_help(script):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], check=False, capture_output=True, text=True)
    assert r.returncode == 0 and "--out" in r.stdout


def test_refuses_test_seeds(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_baselines.py"), "--seeds", "5000", "2", "--out", str(tmp_path)],
        check=False, capture_output=True, text=True,
    )
    assert r.returncode == 2 and "test range" in r.stderr
