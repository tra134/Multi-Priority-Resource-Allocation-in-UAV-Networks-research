"""Run manifest: what produced a results file (CLAUDE.md Rule 4).

Every run appends one JSON line with the config, git commit, seeds, timestamp
and LLM client type, so a number in a table can be traced to its inputs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def _git(args, cwd):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def sha256_of(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def collect_manifest(root, cfg, client, args=None, files=()):
    """Build the manifest dict.  ``client`` is "none", "mock" or "real"."""
    root = str(root)
    dirty = _git(["status", "--porcelain", "--untracked-files=no"], root)
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "config_sha256": sha256_of(cfg),
        "file_sha256": {str(f): file_sha256(Path(root) / f) for f in files},
        "client": client,
        "python": platform.python_version(),
        "argv": list(sys.argv),
        "args": dict(args or {}),
    }


def append_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, sort_keys=True, default=str) + "\n")
