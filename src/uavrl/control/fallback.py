"""Hard-coded safe fallback allocation (Phase 6).

Emergency gets 0.8 bandwidth and 0.8 power; the remaining 0.2 of each is split
0.6 / 0.4 between Important and Normal.
"""

from __future__ import annotations

import numpy as np

FALLBACK_BW = np.array([0.8, 0.2 * 0.6, 0.2 * 0.4], dtype=float)   # [0.8, 0.12, 0.08]
FALLBACK_POWER = np.array([0.8, 0.12, 0.08], dtype=float)


def fallback_shares():
    """Return (bw_share, power_share) for the safe fallback."""
    return FALLBACK_BW.copy(), FALLBACK_POWER.copy()


def fallback_action():
    """Return the raw 6-dim action corresponding to the fallback shares."""
    from .mask import _softmax_inv

    b = _softmax_inv(FALLBACK_BW)
    p = _softmax_inv(FALLBACK_POWER)
    return np.concatenate([b, p]).astype(np.float32)
