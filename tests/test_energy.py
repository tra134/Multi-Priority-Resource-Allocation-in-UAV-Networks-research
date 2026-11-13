import numpy as np

from uavmec.env import energy


def test_eq2_energy_per_cycle():
    assert np.isclose(energy.energy_per_cycle(1e-27, 1e9), 1e-9)


def test_eq2_local_energy_and_cap():
    req = energy.local_energy_required(1e6, 23.0, 1e-27, 1e9)  # 23e6 cycles * 1e-9
    assert np.isclose(req, 0.023)
    assert np.isclose(energy.local_energy(req, 1.0), 0.023)
    assert np.isclose(energy.local_energy(req, 0.01), 0.01)  # min{., E_n}


def test_eq8_offload_energy():
    assert np.isclose(energy.offload_energy_required(0.5, 0.01, 1.0), 0.005)
    # transmit time capped at one slot, finite for zero rate
    assert np.isclose(energy.offload_energy_required(0.5, np.inf, 1.0), 0.5)


def test_eq10_battery_update():
    out = energy.battery_update(np.array([1.0, 0.001]), np.array([0.25, 0.5]))
    assert np.allclose(out, [0.75, 0.0])


def test_a37_uav_hover_energy():
    from uavmec.config import load_config

    cfg = load_config()
    assert cfg.uav_hover_power_w == 168.4842
    out = energy.uav_hover_energy(cfg.n_uavs, cfg.uav_hover_power_w, cfg.slot_s)
    assert out.shape == (cfg.n_uavs,)
    assert np.allclose(out, 168.4842 * cfg.slot_s)
    assert not hasattr(energy, "P0")
