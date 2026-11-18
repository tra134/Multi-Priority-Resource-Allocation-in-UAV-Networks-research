"""Configuration of the Lin et al. 2025 environment (docs/lin2025_spec.md, Sections 1, 5, 6)."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
BANDWIDTH_SHARING = ("independent", "proportional")  # "proportional" = A-42 (lin-env-v1.1)


@dataclass(frozen=True)
class LinConfig:
    """All parameters of the environment. Source tags are in configs/lin2025.yaml."""

    n_uavs: int = 2
    n_sensors: int = 8
    n_subcarriers: int = 4
    bandwidth_hz: float = 250.0e6
    area_m: float = 100.0
    uav_height_m: float = 100.0
    uav_hover_power_w: float = 168.4842
    uav_v_max: float = 50.0
    slot_s: float = 1.0
    n_slots: int = 10
    p_max_w: float = 1.0
    f_sensor_hz: float = 1.0e9
    f_uav_hz: float = 1.4e9
    sensor_battery_j: tuple = (100.0e3, 400.0e3)
    uav_battery_j: tuple = (100.0e3, 500.0e3)
    rate_req_bps: float = 10.0e6
    noise_w: float = 1.0e-10
    beta0: float = 1.0e-5
    task_unit_bits: float = 50.0e3
    task_poisson_lambda: float = 16.0
    task_min_bits: float = 100.0e3
    task_max_bits: float = 1.5e6
    deadline_min_s: float = 0.001
    deadline_max_s: float = 0.050
    c_cyc: float = 24.5  # C-1: calibrated in test T-1
    kappa: float = 1.0009e-27  # C-2: calibrated in test T-3 (A-34)
    d_safe_m: float = 10.0
    reward_success: float = 3.0
    reward_fail: float = -1.0
    # "independent": B_n = f_n B (lin-env-v1.0). "proportional": A-42 (lin-env-v1.1), see lin_env.py.
    bandwidth_sharing: str = "independent"

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def subcarrier_bw_hz(self) -> float:
        """B_m in Hz: 250 MHz per subcarrier (A-3, Table 1 and Section 6.1)."""
        return self.bandwidth_hz


def _validate(c: LinConfig) -> None:
    for name in ("n_uavs", "n_sensors", "n_subcarriers", "n_slots"):
        v = getattr(c, name)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    positive = (
        "bandwidth_hz", "area_m", "uav_height_m", "uav_v_max", "slot_s", "p_max_w",
        "f_sensor_hz", "f_uav_hz", "rate_req_bps", "noise_w", "beta0", "task_unit_bits",
        "task_poisson_lambda", "task_min_bits", "task_max_bits", "deadline_min_s",
        "deadline_max_s", "c_cyc", "kappa",
        "uav_hover_power_w",
    )
    for name in positive:
        if not getattr(c, name) > 0:
            raise ValueError(f"{name} must be > 0, got {getattr(c, name)!r}")
    if c.d_safe_m < 0:
        raise ValueError("d_safe_m must be >= 0")
    for name in ("sensor_battery_j", "uav_battery_j"):
        lo, hi = getattr(c, name)
        if not 0 < lo <= hi:
            raise ValueError(f"{name} must satisfy 0 < low <= high, got {(lo, hi)!r}")
    if c.task_min_bits > c.task_max_bits:
        raise ValueError("task_min_bits must be <= task_max_bits")
    if c.deadline_min_s > c.deadline_max_s:
        raise ValueError("deadline_min_s must be <= deadline_max_s")
    if c.reward_success <= c.reward_fail:
        raise ValueError("reward_success must be larger than reward_fail")
    if c.bandwidth_sharing not in BANDWIDTH_SHARING:
        raise ValueError(f"bandwidth_sharing must be one of {BANDWIDTH_SHARING}, got {c.bandwidth_sharing!r}")


def load_config(path: str | Path = "configs/lin2025.yaml", **overrides) -> LinConfig:
    """Load the YAML config and apply keyword overrides. Raises ValueError on bad values."""
    p = Path(path)
    if not p.exists() and not p.is_absolute() and (_REPO_ROOT / p).exists():
        p = _REPO_ROOT / p
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(LinConfig)}
    unknown = (set(raw) | set(overrides)) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    raw.update(overrides)
    for key, val in list(raw.items()):
        if isinstance(val, list):
            raw[key] = tuple(val)
    cfg = LinConfig(**raw)
    return replace(cfg)
