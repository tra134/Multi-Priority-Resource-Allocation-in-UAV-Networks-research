"""Safety mask: bandwidth floors + emergency power floor with TTL.

Projection guarantees:
    b = f + (1 - sum(f)) * b_raw      -> b >= f, sum(b) = 1, requires sum(f) <= 1
    p = f_p + (1 - sum(f_p)) * p_raw  -> same for power

An invalid mask (sum(f) > 1 or any floor out of [0,1]) is rejected and treated
as an invalid LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Mask:
    """An active safety mask with a TTL (expires_at = slot when it lapses)."""

    bw_floor: np.ndarray          # shape (3,), per-class bandwidth floor [0,1]
    power_floor_emg: float        # emergency power floor [0,1]
    expires_at: int               # slot index when the mask expires

    def valid(self) -> bool:
        f = np.asarray(self.bw_floor, dtype=float)
        return (
            bool(np.all(f >= 0.0)) and bool(np.all(f <= 1.0))
            and bool(0.0 <= self.power_floor_emg <= 1.0)
            and bool(f.sum() <= 1.0 + 1e-9)
        )

    def expired(self, slot: int) -> bool:
        return slot >= self.expires_at


def project_bw_power(b_raw, p_raw, mask: Mask | None):
    """Project raw bandwidth/power shares through an active mask.

    Parameters
    ----------
    b_raw, p_raw : array-like shape (3,)
        Raw shares from softmax (each sums to 1).
    mask : Mask or None
        If None, returns raw shares unchanged.

    Returns
    -------
    (b, p) projected shares, each summing to 1 and respecting floors.

    Raises
    ------
    ValueError if the mask is invalid.
    """
    b_raw = np.asarray(b_raw, dtype=float)
    p_raw = np.asarray(p_raw, dtype=float)
    if mask is None:
        return b_raw.copy(), p_raw.copy()
    if not mask.valid():
        raise ValueError("invalid mask: floors out of range or sum(bw_floor) > 1")

    f_bw = np.asarray(mask.bw_floor, dtype=float)
    f_p = np.array([mask.power_floor_emg, 0.0, 0.0], dtype=float)

    b = f_bw + (1.0 - f_bw.sum()) * b_raw
    p = f_p + (1.0 - f_p.sum()) * p_raw
    return b, p


def project_action(action, mask: Mask | None):
    """Project a 6-dim raw action (in [-1,1]) through a mask.

    Returns the projected 6-dim action (inverse softmax of projected shares) so
    it can be fed back through the same softmax(3*x) mapping, OR the projected
    shares directly.  Here we return projected shares (b, p) for the env to use.
    """
    b_raw = _softmax_inv(action[:3])
    p_raw = _softmax_inv(action[3:])
    b, p = project_bw_power(b_raw, p_raw, mask)
    return b, p


def _softmax_inv(shares):
    shares = np.clip(np.asarray(shares, dtype=float), 1e-9, 1.0)
    log_s = np.log(shares)
    return (log_s - np.mean(log_s)) / 3.0
