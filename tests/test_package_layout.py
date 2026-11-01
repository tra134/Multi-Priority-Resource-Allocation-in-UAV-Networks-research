"""Guard against .gitignore patterns hiding source packages.

The bare ``env/`` virtualenv pattern used to ignore ``src/uavrl/env/``, so
GitHub clones (and therefore Colab) were missing the environment package
entirely and failed with ``ModuleNotFoundError: uavrl.env``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_source_file_is_gitignored():
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py"))
    assert files, "src/ contains no Python files"

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        # Not a Git checkout (e.g. a source distribution): nothing to guard.
        return

    assert result.returncode == 1, "gitignored source files:\n" + result.stdout


def test_env_package_is_importable():
    from uavrl.env.scenarios import Scenario
    from uavrl.env.uav_env import UAVEnv

    assert UAVEnv is not None
    assert Scenario is not None
