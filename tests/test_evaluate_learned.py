"""Tests for scripts/evaluate_learned.py (unit-test seeds 0-99 only, never test seeds 5000+)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN = str(ROOT / "scripts" / "train.py")
SCRIPT = str(ROOT / "scripts" / "evaluate_learned.py")
SMALL = ["--n-sensors", "2", "--episodes", "2", "--set", "eval_episodes=1", "--set", "eval_every=1000",
         "--set", "ckpt_every=2", "--set", "warmup=2", "--set", "batch=4", "--set", "rollout_episodes=1",
         "--set", "minibatch=4", "--set", "epochs=1"]


def sh(script, *args):
    return subprocess.run([sys.executable, script, *args], check=False, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("runs")
    specs = {
        "md_sic": ["--algo", "maddpg", "--mode", "POS"],
        "md_nosic": ["--algo", "maddpg", "--mode", "POS", "--no-sic"],
        "mp_sic": ["--algo", "mappo", "--mode", "POS"],
    }
    out = {}
    for name, a in specs.items():
        d = base / name
        r = sh(TRAIN, *a, *SMALL, "--out", str(d))
        assert r.returncode == 0, r.stderr
        out[name] = d
    return out


@pytest.fixture(scope="module")
def evaluated(runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    r = sh(SCRIPT, "--runs", *map(str, runs.values()), "--out", str(out), "--seeds-start", "0", "--n-episodes", "2")
    assert r.returncode == 0, r.stderr
    return out


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_help():
    r = sh(SCRIPT, "--help")
    assert r.returncode == 0 and "--runs" in r.stdout and "--out" in r.stdout


def test_out_required():
    r = sh(SCRIPT, "--runs", "x")
    assert r.returncode != 0 and "--out" in r.stderr


def test_files_and_columns(evaluated):
    for n in ("per_run.csv", "summary.csv", "gate.json", "manifest.json"):
        assert (evaluated / n).exists(), n
    pr = read(evaluated / "per_run.csv")
    assert len(pr) == 4  # 3 learned runs + Local_Computing at N = 2
    for c in ("method", "n_sensors", "seed", "run_dir", "success_rate", "mean_latency_ms", "mean_energy_mw",
              "n_episodes", "git_commit"):
        assert c in pr[0], c
    assert {r["method"] for r in pr} == {"POS_MADDPG_NOMA", "POS_MADDPG_noSIC", "POS_MAPPO_NOMA", "Local_Computing"}
    assert all(r["n_episodes"] == "2" for r in pr)


def test_summary_has_local(evaluated):
    rows = read(evaluated / "summary.csv")
    methods = {r["method"] for r in rows}
    assert "Local_Computing" in methods and "POS_MADDPG_NOMA" in methods
    for c in ("n_seeds", "success_rate_mean", "success_rate_std", "mean_latency_ms_mean", "mean_energy_mw_mean",
              "target_success_rate", "target_note"):
        assert c in rows[0], c
    # targets only exist for N = 8; this run uses N = 2
    assert all(r["target_success_rate"] == "" for r in rows)


def test_gate_json(evaluated):
    g = json.loads((evaluated / "gate.json").read_text())
    items = g.get("items", g)
    assert len(items) == 4
    for it in items.values():
        assert it["status"] in ("pass", "fail", "not_evaluable") and "reason" in it
    assert items["3"]["status"] == "not_evaluable"
    assert items["1"]["status"] == "not_evaluable"  # missing methods and N != 8


def test_manifest(evaluated, runs):
    m = json.loads((evaluated / "manifest.json").read_text())
    assert m["llm_client"] == "none" and "git_commit" in m and "git_dirty" in m
    assert m["seeds"] == [0, 1] and len(m["run_dirs"]) == 3


def test_missing_final_pt(runs, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(runs["md_sic"], bad)
    (bad / "final.pt").unlink()
    r = sh(SCRIPT, "--runs", str(bad), "--out", str(tmp_path / "o"), "--seeds-start", "0", "--n-episodes", "1")
    assert r.returncode != 0 and "final.pt" in r.stderr
