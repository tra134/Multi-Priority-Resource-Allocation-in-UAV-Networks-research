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


# ---- A-38: gate item 3 on synthetic per-seed data (no training) ----
NS = (2, 4, 6, 8, 10, 12)
M3 = ("POS_MADDPG_NOMA", "FOS_MADDPG_NOMA", "POS_MAPPO_NOMA")


def _load_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_learned", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(mod)
    return mod.gate


def _synth(succ_slope_pp=-3.0, lat_slope=0.9, base=0.95, seeds=(0, 1, 2), ns=NS, methods=M3,
           overrides=None, seed_sets=None, fos_bonus=0.02):
    """Return (summ, per_seed) with linear per-seed trends; overrides[(method, seed)] = (pp slope, ms slope)."""
    overrides = overrides or {}
    per_seed, summ = {}, {}
    for m in methods:
        for n in ns:
            ss = (seed_sets or {}).get((m, n), seeds)
            d = {}
            for s in ss:
                ssl, lsl = overrides.get((m, s), (succ_slope_pp, lat_slope))
                sr = base + (fos_bonus if m == "FOS_MADDPG_NOMA" else 0.0) + ssl / 100.0 * (n - 2)
                d[s] = {"success_rate": sr, "mean_latency_ms": 5.0 + lsl * (n - 2)}
            per_seed[(m, n)] = d
            summ[(m, n)] = {k: sum(v[k] for v in d.values()) / len(d) for k in ("success_rate", "mean_latency_ms")}
    return summ, per_seed


def _item3(**kw):
    summ, per_seed = _synth(**kw)
    return _load_gate()(summ, per_seed)["3"]


def test_a38_pass():
    it = _item3()
    assert it["status"] == "pass", it
    assert it["thresholds"] and "seed_sets" in it and "slopes" in it


def test_a38_too_flat_fails():
    assert _item3(succ_slope_pp=-0.5)["status"] == "fail"


def test_a38_one_seed_positive_fails():
    it = _item3(overrides={("POS_MADDPG_NOMA", 0): (+0.5, 0.9), ("POS_MADDPG_NOMA", 1): (-6.0, 0.9),
                           ("POS_MADDPG_NOMA", 2): (-6.0, 0.9)})
    assert it["status"] == "fail"  # mean slope -3.8 passes, but one seed slope is positive


def test_a38_missing_n_not_evaluable():
    assert _item3(ns=(2, 4, 6, 8, 10))["status"] == "not_evaluable"


def test_a38_seed_sets_differ_not_evaluable():
    assert _item3(seed_sets={("POS_MAPPO_NOMA", 12): (0, 1, 3)})["status"] == "not_evaluable"


def test_a38_missing_method_not_evaluable():
    assert _item3(methods=M3[:2])["status"] == "not_evaluable"


def test_a38_latency_threshold():
    assert _item3(lat_slope=0.2)["status"] == "fail"
    assert _item3(lat_slope=0.31)["status"] == "pass"
    assert _item3(overrides={("FOS_MADDPG_NOMA", 1): (-3.0, -0.1)})["status"] == "fail"


def test_a38_fos_best_fail():
    assert _item3(fos_bonus=-0.02)["status"] == "fail"


def test_a41_n8_extra_seeds_use_sweep_seeds_only():
    """A-41: at N=8 the Table 2 runs have seeds 0-4, the sweep has seeds 0-2; item 3 uses seeds 0-2 at N=8."""
    ss = {(m, 8): (0, 1, 2, 3, 4) for m in M3}
    it = _item3(seed_sets=ss)
    assert it["status"] == "pass", it
    for m in M3:
        assert sorted(it["slopes"][m]["success_pp"]["seeds"]) == [0, 1, 2]


def test_a38_n8_missing_sweep_seed_not_evaluable():
    """A-38: a sweep seed (0-2) missing at N=8 still gives not_evaluable."""
    assert _item3(seed_sets={("POS_MAPPO_NOMA", 8): (0, 1, 3, 4)})["status"] == "not_evaluable"
    assert _item3(seed_sets={("POS_MAPPO_NOMA", 6): (0, 1)})["status"] == "not_evaluable"


def test_recompute_gate_from_per_run(tmp_path):
    """scripts/recompute_gate.py rebuilds the gate from a per_run.csv without any evaluation."""
    cols = ["method", "n_sensors", "seed", "run_dir", "success_rate", "mean_latency_ms", "mean_energy_mw",
            "n_episodes", "git_commit"]
    rows = []
    for m in M3:
        for n in NS:
            for s in ((0, 1, 2, 3, 4) if n == 8 else (0, 1, 2)):
                rows.append([m, n, s, "x", 0.95 - 0.03 * (n - 2), 5.0 + 0.9 * (n - 2), 1.0, 100, "abc"])
    csv_in = tmp_path / "per_run.csv"
    csv_in.write_text(",".join(cols) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    out = tmp_path / "out"
    r = sh(str(ROOT / "scripts" / "recompute_gate.py"), "--per-run", str(csv_in), "--out", str(out))
    assert r.returncode == 0, r.stderr
    g = json.loads((out / "gate.json").read_text())
    assert g["items"]["3"]["status"] == "pass"
    man = json.loads((out / "manifest.json").read_text())
    assert man["llm_client"] == "none" and len(man["input_sha256"]) == 64
    assert sh(str(ROOT / "scripts" / "recompute_gate.py"), "--per-run", str(csv_in), "--out", str(out)).returncode != 0  # no overwrite
