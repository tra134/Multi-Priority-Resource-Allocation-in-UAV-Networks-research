"""Energy models (Eq. 2, 8, 10, 17 of Diao; assumptions A-11, A-37)."""

from __future__ import annotations

import numpy as np


def energy_per_cycle(kappa: float, f_hz: float) -> float:
    """Eq. 2: z_n = kappa * F_n^2 (J per CPU cycle)."""
    return kappa * f_hz**2


def local_energy_required(bits: np.ndarray, c_cyc: float, kappa: float, f_hz: float) -> np.ndarray:
    """Energy the local computation needs: z_n * C_n * c_cyc (J), before the battery cap."""
    return energy_per_cycle(kappa, f_hz) * np.asarray(bits, dtype=float) * c_cyc


def local_energy(required_j: np.ndarray, battery_j: np.ndarray) -> np.ndarray:
    """Eq. 2: e^l_n = min{ z_n * C_n * c_cyc, E_n(t) } for a locally computed task (J)."""
    return np.minimum(required_j, battery_j)


def offload_energy_required(power_w: np.ndarray, t_off_s: np.ndarray, slot_s: float) -> np.ndarray:
    """Eq. 8 offloading energy e^0_n = p_n * T^0_n for an offloading user (J). # A-11 (T-4)

    The transmit time is capped at one slot (a transmission cannot outlast its slot), which also
    keeps the energy finite when the rate is zero. # A-18: cap not in the paper.
    """
    return np.asarray(power_w, dtype=float) * np.minimum(np.asarray(t_off_s, dtype=float), slot_s)


def battery_update(battery_j: np.ndarray, spent_j: np.ndarray) -> np.ndarray:
    """Eq. 10: E_n(t+1) = E_n(t) - e^0_n - e^l_n (spent_j is the sum), never below 0."""
    return np.maximum(battery_j - spent_j, 0.0)


def uav_hover_energy(n_uavs: int, hover_power_w: float, slot_s: float) -> np.ndarray:
    """Hover energy of one slot per UAV: E = P_H * time (J). # A-37: Diao et al. 2021 Eq. 17."""
    return np.full(n_uavs, hover_power_w * slot_s)
