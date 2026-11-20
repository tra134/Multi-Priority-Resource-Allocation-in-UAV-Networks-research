"""Air-to-ground channel (assumption A-1)."""

from __future__ import annotations

import numpy as np


def channel_gain(
    sensor_pos: np.ndarray,
    uav_pos: np.ndarray,
    height_m: float,
    beta0: float,
    extra_loss_db: float = 0.0,
) -> np.ndarray:
    """Power gain |h_nk|^2 = beta0 / (H^2 + ||q_k - w_n||^2) (A-1, free-space LoS).

    sensor_pos: (N, 2) ground positions in m. uav_pos: (K, 2) in m. Returns (N, K) linear gain.
    beta0 is the gain at 1 m (-50 dB = 1e-5).
    extra_loss_db (A-44, lin-env-v1.2): uniform extra link loss L, gain multiplied by 10^(-L/10).
    L = 0 returns the v1.0 gain unchanged (bit-identical).
    """
    diff = sensor_pos[:, None, :] - uav_pos[None, :, :]
    d2 = np.sum(diff**2, axis=-1)
    gain = beta0 / (height_m**2 + d2)
    if extra_loss_db != 0.0:  # A-44
        gain = gain * 10.0 ** (-extra_loss_db / 10.0)  # A-44
    return gain
