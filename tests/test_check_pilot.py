"""Tests for scripts/check_pilot.py (A-39 pilot rule) on synthetic logs; no training, no paper numbers."""

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "scripts" / "check_pilot.py")


def sh(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], check=False, capture_output=True, text=True, cwd=ROOT)


def make_run(base, name, succ, episodes=10000, every=500, losses=None, commit="abc123"):
    """succ: validation success (fraction) per evaluation, at episodes every, 2*every, ..."""
    d = base / name
    d.mkdir()
    with (d / "eval_log.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "success_rate", "mean_latency_ms", "mean_energy_mw", "n_episodes"])
        for i, s in enumerate(succ):
            w.writerow([(i + 1) * every, s, 8.0, 5.0, 20])
    with (d / "train_log.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward_sum", "success_rate", "mean_latency_ms", "mean_energy_mw", "losses"])
        for ep in range(5):
            w.writerow([ep, 1.0, 0.5, 8.0, 5.0, json.dumps(losses or {"critic": 0.1, "actor": -0.2})])
    (d / "manifest.json").write_text(json.dumps({"git_commit": commit, "cli_args": {"algo": "maddpg"},
                                                 "train_config": {"episodes": episodes}}))
    return d


def flat(n=20, v=0.80):
    return [v] * n


def run_check(tmp_path, *dirs):
    out = tmp_path / "out"
    r = sh(*[a for d in dirs for a in ("--run", str(d))], "--out", str(out))
    return r, out


def result(out):
    return json.loads((out / "pilot_check.json").read_text())


def test_converged_pass(tmp_path):
    r, out = run_check(tmp_path, make_run(tmp_path, "a", flat()))
    assert r.returncode == 0, r.stderr
    j = result(out)
    run = next(iter(j["runs"].values())) if isinstance(j["runs"], dict) else j["runs"][0]
    assert run["T1"] == "pass" and run["T2"] == "pass" and run["decision"] == "lock"
    assert j["decision"] == "lock"
    assert abs(run["Delta"]) < 1e-9
    assert run["eval_episodes_last"] == [9000, 9500, 10000]
    assert run["eval_episodes_prev"] == [7500, 8000, 8500]
    m = json.loads((out / "manifest.json").read_text())
    assert "git_commit" in m and "git_dirty" in m and "timestamp" in m and "cli_args" in m
    assert "abc123" in json.dumps(m)


def test_delta_plus3_extends(tmp_path):
    s = flat(17, 0.70) + flat(3, 0.73)
    # prev window = 3 evals before last three; last three = 0.73, prev = 0.70 -> Delta = +3 pp
    _, out = run_check(tmp_path, make_run(tmp_path, "a", s))
    j = result(out)
    assert j["decision"] == "extend_to_20000"


def test_delta_minus3_with_t2_pass_extends(tmp_path):
    s = flat(17, 0.80) + flat(3, 0.77)
    _, out = run_check(tmp_path, make_run(tmp_path, "a", s))
    assert result(out)["decision"] == "extend_to_20000"


def test_best_minus_last_6pp_stops(tmp_path):
    s = flat(20, 0.80)
    s[10] = 0.86  # best 0.86, last 0.80 -> 6 pp; T1 windows unaffected
    _, out = run_check(tmp_path, make_run(tmp_path, "a", s))
    assert result(out)["decision"] == "stop_report"


def test_nan_loss_stops(tmp_path):
    d = make_run(tmp_path, "a", flat(), losses={"critic": float("nan"), "actor": 0.0})
    _, out = run_check(tmp_path, d)
    j = result(out)
    assert j["decision"] == "stop_report"
    run = next(iter(j["runs"].values())) if isinstance(j["runs"], dict) else j["runs"][0]
    assert run["nan_findings"]


def test_20000_episodes_t1_fail_stops(tmp_path):
    s = flat(37, 0.70) + flat(3, 0.73)
    d = make_run(tmp_path, "a", s, episodes=20000)
    _, out = run_check(tmp_path, d)
    j = result(out)
    assert j["decision"] == "stop_report"
    assert "at most one extension" in json.dumps(j).lower()


def test_overall_decision_combination(tmp_path):
    a = make_run(tmp_path, "a", flat())
    b = make_run(tmp_path, "b", flat(17, 0.70) + flat(3, 0.73))
    _, out = run_check(tmp_path, a, b)
    assert result(out)["decision"] == "extend_to_20000"


def test_out_required(tmp_path):
    r = sh("--run", str(make_run(tmp_path, "a", flat())))
    assert r.returncode != 0 and "--out" in r.stderr


def test_too_few_evaluations_exit_2(tmp_path):
    r, _ = run_check(tmp_path, make_run(tmp_path, "a", flat(5)))
    assert r.returncode == 2 and "evaluations" in r.stderr
