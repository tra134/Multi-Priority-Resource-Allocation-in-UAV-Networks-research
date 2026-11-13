"""Tests for scripts/train.py (integration tests skip until the learners exist)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "scripts" / "train.py")
SMALL = ["--n-sensors", "2", "--set", "eval_episodes=1", "--set", "eval_every=3", "--set", "ckpt_every=2",
         "--set", "warmup=2", "--set", "batch=4", "--set", "rollout_episodes=1", "--set", "minibatch=4",
         "--set", "epochs=1"]


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], check=False, capture_output=True, text=True, cwd=ROOT)


def test_help():
    r = run("--help")
    assert r.returncode == 0 and "--out" in r.stdout


def test_out_required():
    r = run("--algo", "maddpg")
    assert r.returncode != 0 and "--out" in r.stderr


@pytest.mark.parametrize("algo", ["maddpg", "mappo"])
def test_smoke(algo, tmp_path):
    pytest.importorskip(f"uavmec.agents.{algo}")
    r = run("--algo", algo, "--episodes", "3", "--out", str(tmp_path), *SMALL)
    assert r.returncode == 0, r.stderr
    for name in ("train_log.csv", "eval_log.csv", "manifest.json", "ckpt.pt", "final.pt"):
        assert (tmp_path / name).exists(), name
    assert len((tmp_path / "train_log.csv").read_text().splitlines()) == 4
    assert len((tmp_path / "eval_log.csv").read_text().splitlines()) == 2


@pytest.mark.parametrize("algo", ["maddpg", "mappo"])
def test_resume_matches_straight_run(algo, tmp_path):
    pytest.importorskip(f"uavmec.agents.{algo}")
    args = ["--algo", algo, *SMALL, "--set", "eval_every=1000"]
    a, b = tmp_path / "a", tmp_path / "b"
    assert run(*args, "--episodes", "4", "--out", str(a)).returncode == 0
    assert run(*args, "--episodes", "2", "--out", str(b)).returncode == 0
    r = run(*args, "--episodes", "4", "--out", str(b), "--resume")
    assert r.returncode == 0, r.stderr
    assert (a / "train_log.csv").read_text() == (b / "train_log.csv").read_text()
