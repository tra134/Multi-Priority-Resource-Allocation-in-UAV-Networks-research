"""Local computing latency (Eq. 1, assumption A-4)."""

from __future__ import annotations

import numpy as np


def local_latency(bits: np.ndarray, c_cyc: float, f_hz: float) -> np.ndarray:
    """Eq. 1 with A-4: T^l_n = C_n * c_cyc / F_n (seconds).

    The paper writes D_n / F_n; D_n is the latency requirement, so the CPU-cycle
    count C_n * c_cyc is used instead.
    """
    return np.asarray(bits, dtype=float) * c_cyc / f_hz
