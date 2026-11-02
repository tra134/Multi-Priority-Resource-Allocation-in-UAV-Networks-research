"""Tests for dotenv/YAML configuration helpers (Phase 8 setup)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from uavrl.config import (
    checkpoint_run_name,
    find_project_root,
    llm_settings,
    load_env,
)


def test_load_env_does_not_override_by_default(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A_TEST_VALUE=from_file\n# comment\nB_TEST_VALUE='quoted'\n")
    monkeypatch.setenv("A_TEST_VALUE", "from_shell")
    monkeypatch.delenv("B_TEST_VALUE", raising=False)

    values = load_env(env_file)

    assert values["A_TEST_VALUE"] == "from_file"
    assert values["B_TEST_VALUE"] == "quoted"
    assert os.environ["A_TEST_VALUE"] == "from_shell"


def test_load_env_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("A_TEST_VALUE=from_file\n")
    monkeypatch.setenv("A_TEST_VALUE", "from_shell")
    load_env(env_file, override=True)
    assert os.environ["A_TEST_VALUE"] == "from_file"


def test_llm_settings_env_wins(tmp_path, monkeypatch):
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(yaml.safe_dump({
        "llm": {"base_url": "http://yaml", "model": "yaml-model", "api_key": None},
        "cache": {"enabled": False, "dir": "x"},
    }))
    # load_env() inside llm_settings looks for the project-root .env; point the
    # relevant vars explicitly instead so this test is hermetic.
    monkeypatch.setenv("LLM_BASE_URL", "http://env")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    # Prevent the real repo .env from touching this test.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uavrl.config.find_project_root", lambda *a, **k: tmp_path)

    llm = llm_settings(cfg)
    assert llm["base_url"] == "http://env"
    assert llm["model"] == "env-model"
    assert llm["api_key"] == "env-key"
    assert llm["cache"]["enabled"] is False


def test_find_project_root_contains_pyproject():
    root = find_project_root()
    assert (Path(root) / "pyproject.toml").exists()


def test_checkpoint_run_name_matches_grid_layout():
    """train.py and run_grid.py must agree on the checkpoint filename."""
    assert checkpoint_run_name("sac", 0, 10) == "sac_seed0_N10"
    assert checkpoint_run_name("sac", 3, 2, dr_wide=True) == \
        "sac_seed3_N2_drwide"
