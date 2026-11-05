"""Run manifest records the provenance of results (CLAUDE.md Rule 4)."""

import json
from pathlib import Path

from uavrl.eval.manifest import append_manifest, collect_manifest, sha256_of

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_provenance_fields(tmp_path):
    m = collect_manifest(ROOT, {"a": 1}, client="mock", args={"seeds": [0]},
                         files=["configs/default.yaml"])
    for key in ("timestamp_utc", "git_commit", "git_dirty", "config_sha256",
                "client", "file_sha256", "argv"):
        assert key in m
    assert m["client"] == "mock"
    assert m["file_sha256"]["configs/default.yaml"]
    path = tmp_path / "manifest.jsonl"
    append_manifest(path, m)
    append_manifest(path, m)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["client"] == "mock"


def test_config_hash_is_order_independent():
    assert sha256_of({"a": 1, "b": 2}) == sha256_of({"b": 2, "a": 1})
    assert sha256_of({"a": 1}) != sha256_of({"a": 2})
