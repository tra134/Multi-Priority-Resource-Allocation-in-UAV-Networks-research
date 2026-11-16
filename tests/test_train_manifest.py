"""P-013 / P-015: train.py must keep the history of every start/resume (manifest_history.jsonl, append only)
and write manifest.json, logs and checkpoints atomically, also when Colab is cut off mid-run."""

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_train import ROOT, SCRIPT, SMALL, run

pytest.importorskip("uavmec.agents.maddpg")
ARGS = ["--algo", "maddpg", *SMALL, "--set", "eval_every=1000"]


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def history(out: Path) -> list[dict]:
    recs, bad = load_train_module().read_history(out / "manifest_history.jsonl")
    assert bad == 0
    return recs


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_start_then_resume_keeps_both_segments(tmp_path):
    assert run(*ARGS, "--episodes", "2", "--out", str(tmp_path)).returncode == 0
    ck_sha = sha256(tmp_path / "ckpt.pt")
    r = run(*ARGS, "--episodes", "4", "--out", str(tmp_path), "--resume")
    assert r.returncode == 0, r.stderr
    recs = history(tmp_path)
    starts = [x for x in recs if x["event"] in ("start", "resume")]
    assert [x["event"] for x in starts] == ["start", "resume"]
    s0, s1 = starts
    assert (s0["segment"], s0["start_episode"], s0["planned_end_episode"]) == (0, 0, 2)
    assert s0["source_checkpoint"] is None
    assert (s1["segment"], s1["start_episode"], s1["planned_end_episode"]) == (1, 2, 4)
    assert s1["source_checkpoint"]["episode"] == 2
    assert s1["source_checkpoint"]["sha256"] == ck_sha
    for x in starts:
        assert len(x["git_commit"]) >= 7 and "timestamp" in x and "config_sha256" in x
    assert s0["config_sha256"] == s1["config_sha256"]  # same settings in both segments
    ends = [(x["segment"], x["end_episode"]) for x in recs if x["event"] == "end"]
    assert ends == [(0, 2), (1, 4)]
    ckpts = [(x["segment"], x["episode"]) for x in recs if x["event"] == "checkpoint"]
    assert ckpts == [(0, 2), (1, 4)]
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["resumed_from_episode"] == 2 and man["segment"] == 1
    assert man["history_file"] == "manifest_history.jsonl"


def test_truncated_last_line_is_isolated(tmp_path):
    """A crash in the middle of an append leaves a partial line; the next append must not merge with it."""
    assert run(*ARGS, "--episodes", "2", "--out", str(tmp_path)).returncode == 0
    h = tmp_path / "manifest_history.jsonl"
    with h.open("a") as f:
        f.write('{"event": "checkpoint", "segm')  # partial line, no newline
    assert run(*ARGS, "--episodes", "4", "--out", str(tmp_path), "--resume").returncode == 0
    recs, bad = load_train_module().read_history(h)
    assert bad == 1
    assert [x["event"] for x in recs if x["event"] in ("start", "resume")] == ["start", "resume"]
    assert recs[-1] == {**recs[-1], "event": "end", "end_episode": 4}


def test_killed_run_then_resume(tmp_path):
    """Hard kill (like a Colab disconnect) after a checkpoint; resume continues from the last checkpoint
    and the history shows where the killed segment stopped (no 'end' record for it)."""
    proc = subprocess.Popen([sys.executable, SCRIPT, *ARGS, "--episodes", "100000", "--out", str(tmp_path)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    h = tmp_path / "manifest_history.jsonl"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if h.exists() and h.read_text().count('"checkpoint"') >= 2:
                break
            time.sleep(0.2)
        else:
            pytest.fail("no checkpoint written in time")
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    ck_ep = int(load_train_module().torch_load_episode(tmp_path / "ckpt.pt"))
    r = run(*ARGS, "--episodes", str(ck_ep + 2), "--out", str(tmp_path), "--resume")
    assert r.returncode == 0, r.stderr
    recs = history(tmp_path)
    seg0 = [x for x in recs if x["segment"] == 0]
    assert seg0[0]["event"] == "start" and not any(x["event"] == "end" for x in seg0)
    assert max(x["episode"] for x in seg0 if x["event"] == "checkpoint") >= ck_ep
    res = [x for x in recs if x["event"] == "resume"]
    assert len(res) == 1 and res[0]["start_episode"] == ck_ep and res[0]["segment"] == 1
    # P-015: no gap and no duplicate in the training log after kill + resume
    eps = [int(ln.split(",")[0]) for ln in (tmp_path / "train_log.csv").read_text().splitlines()[1:]]
    assert eps == list(range(ck_ep + 2))


def test_atomic_csv_write_keeps_old_file_on_failure(tmp_path):
    """P-015: a failure while writing a log must not erase the previous log."""
    mod = load_train_module()
    p = tmp_path / "train_log.csv"
    mod.write_rows(p, ["episode"], [{"episode": 0}, {"episode": 1}])
    before = p.read_text()
    with pytest.raises(ValueError):
        mod.write_rows(p, ["episode"], [{"episode": 2}, {"bad_field": 3}])  # DictWriter raises mid-write
    assert p.read_text() == before
    assert os.listdir(tmp_path) == ["train_log.csv"]


def test_fresh_start_in_used_folder_keeps_history(tmp_path):
    """Running again without --resume restarts training but must not erase the old history."""
    assert run(*ARGS, "--episodes", "2", "--out", str(tmp_path)).returncode == 0
    assert run(*ARGS, "--episodes", "2", "--out", str(tmp_path)).returncode == 0
    starts = [x for x in history(tmp_path) if x["event"] in ("start", "resume")]
    assert [(x["event"], x["segment"]) for x in starts] == [("start", 0), ("start", 1)]


def test_atomic_json_write_keeps_old_file_on_failure(tmp_path):
    mod = load_train_module()
    p = tmp_path / "manifest.json"
    mod.write_json_atomic(p, {"a": 1})
    with pytest.raises(TypeError):
        mod.write_json_atomic(p, {"bad": object()})  # serialization fails mid-write
    assert json.loads(p.read_text()) == {"a": 1}
    assert os.listdir(tmp_path) == ["manifest.json"]  # no temp file left behind
