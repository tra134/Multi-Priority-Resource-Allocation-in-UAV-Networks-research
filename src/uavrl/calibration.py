"""Per-checkpoint calibration values (calib-v2, TE-071/TE-073).

``configs/calibration_v2.yaml`` stores, for each checkpoint run name
(for example ``sac_seed1_N10_mask``): ``tau_unc``, ``theta_queue``,
``m8_random_rate`` and ``m9_K``.  ``scripts/run_grid.py`` reads them here so a
stale value in ``configs/default.yaml`` can never be used silently.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# run_grid parameter name -> key in the calibration file
FILE_KEYS = {"tau_unc": "tau_unc", "theta_queue": "theta_queue",
             "random_rate": "m8_random_rate", "periodic_k": "m9_K"}


def load_calibration(path) -> dict:
    """Return ``{run_name: {key: value}}``; a missing file gives ``{}``."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(raw.get("checkpoints") or {})


def required_params(spec) -> tuple:
    """Calibration parameters a method needs (``spec`` from run_grid METHODS)."""
    kind = spec["kind"]
    if kind == "rule":
        return ("theta_queue",)
    if kind != "llm":
        return ()
    trigger = spec["trigger"]
    if trigger == "hybrid":
        return ("tau_unc", "theta_queue")
    if trigger == "random":
        return ("random_rate",)
    if trigger == "periodic":
        return ("periodic_k",)
    return ()


def resolve_params(calib, run_name, overrides, needed):
    """Values and their source ("flag" or "calibration") for one checkpoint.

    An explicit flag (non-None in ``overrides``) wins.  A needed value that is
    missing or null in the file raises ``SystemExit`` with the run name.
    """
    values, sources = {}, {}
    for name in needed:
        if overrides.get(name) is not None:
            values[name], sources[name] = overrides[name], "flag"
            continue
        entry = calib.get(run_name)
        value = None if entry is None else entry.get(FILE_KEYS[name])
        if value is None:
            raise SystemExit(
                f"calibration: no value for {name} ({FILE_KEYS[name]}) of "
                f"{run_name}; set it in configs/calibration_v2.yaml or pass "
                f"--{name.replace('_', '-')}"
            )
        values[name], sources[name] = value, "calibration"
    return values, sources
