"""Air-to-ground channel (assumption A-1)."""

from __future__ import annotations

import numpy as np


def channel_gain(
    sensor_pos: np.ndarray, uav_pos: np.ndarray, height_m: float, beta0: float
) -> np.ndarray:
    """Power gain |h_nk|^2 = beta0 / (H^2 + ||q_k - w_n||^2) (A-1, free-space LoS).

    sensor_pos: (N, 2) ground positions in m. uav_pos: (K, 2) in m. Returns (N, K) linear gain.
    beta0 is the gain at 1 m (-50 dB = 1e-5).
    """
    diff = sensor_pos[:, None, :] - uav_pos[None, :, :]
    d2 = np.sum(diff**2, axis=-1)
    return beta0 / (height_m**2 + d2)
