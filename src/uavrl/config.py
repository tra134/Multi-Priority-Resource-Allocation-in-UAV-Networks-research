"""Small configuration helpers shared by scripts and the LLM client.

The project deliberately keeps secrets out of tracked files.  ``load_env``
reads a local ``.env`` file (already gitignored) and exports its values as
environment variables, without overriding variables that are already set.
No third-party dependency is required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


def find_project_root(start: Optional[os.PathLike | str] = None) -> Path:
    """Walk upwards from ``start`` until a ``pyproject.toml`` is found."""
    here = Path(start or os.getcwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def load_env(path: Optional[os.PathLike | str] = None, override: bool = False) -> dict:
    """Load ``KEY=VALUE`` pairs from a dotenv file into ``os.environ``.

    Returns the values that were read.  Missing files are ignored so that
    Colab / CI runs can configure variables directly in the environment.
    """
    if path is None:
        path = find_project_root() / ".env"
    path = Path(path)
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def load_yaml(path: Optional[os.PathLike | str] = None) -> dict:
    """Load a YAML config file (empty file -> empty dict)."""
    if path is None:
        path = find_project_root() / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def llm_settings(config_path: Optional[os.PathLike | str] = None) -> dict:
    """Return the ``llm`` section of ``configs/llm.yaml`` merged with env vars.

    Environment variables (``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``)
    always win over YAML values, so the same config works on a laptop, on
    Colab, and in CI.
    """
    load_env()
    if config_path is None:
        config_path = find_project_root() / "configs" / "llm.yaml"
    cfg = load_yaml(config_path)
    llm = dict(cfg.get("llm", {}))
    llm["base_url"] = os.environ.get("LLM_BASE_URL") or llm.get("base_url")
    llm["api_key"] = os.environ.get("LLM_API_KEY") or llm.get("api_key")
    llm["model"] = os.environ.get("LLM_MODEL") or llm.get("model")
    llm["cache"] = dict(cfg.get("cache", {}))
    return llm
