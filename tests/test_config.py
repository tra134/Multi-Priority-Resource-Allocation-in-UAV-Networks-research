import dataclasses

import pytest

from uavmec.config import LinConfig, load_config


def test_load_defaults_match_dataclass():
    cfg = load_config()
    assert cfg == LinConfig()
    assert cfg.n_uavs == 2 and cfg.n_sensors == 8 and cfg.n_subcarriers == 4
    assert cfg.bandwidth_hz == 250e6 and cfg.noise_w == 1e-10 and cfg.beta0 == 1e-5
    assert cfg.sensor_battery_j == (100e3, 400e3)
    assert cfg.c_cyc == 24.5 and cfg.kappa == 1.0009e-27


def test_overrides_and_frozen():
    cfg = load_config(n_sensors=4)
    assert cfg.n_sensors == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_sensors = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"n_sensors": 0},
        {"noise_w": -1.0},
        {"sensor_battery_j": (5.0, 1.0)},
        {"task_min_bits": 2e6},
        {"reward_success": -2.0},
        {"unknown_key": 1},
    ],
)
def test_bad_values_raise(bad):
    with pytest.raises(ValueError):
        load_config(**bad)


def test_a3_per_carrier_only():
    """A-3 decided 2026-11-13 (TE-010): 250 MHz per subcarrier; the comparison switch is removed."""
    cfg = load_config()
    assert not hasattr(cfg, "bandwidth_mode")
    assert cfg.subcarrier_bw_hz == 250e6


def test_a11_physical_only():
    """A-11 decided 2026-11-13 (T-4): offloading energy is p * T0; the switch is removed."""
    assert not hasattr(load_config(), "offload_energy")
