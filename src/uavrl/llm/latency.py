"""Simulated LLM latency (finding F13).

Latency used for scheduling must come from the fitted model, not from the
wall clock and not from a cache hit (which returns 0 ms).  This keeps runs
reproducible and the cached and uncached results comparable.  The measured
wall-clock latency of a real call is logged separately.
"""

from __future__ import annotations

import math


class LatencyModel:
    """Lognormal latency [ms] with a given median and log-sigma."""

    def __init__(self, median_ms: float = 1159.0, sigma: float = 0.705):
        if median_ms <= 0 or sigma < 0:
            raise ValueError("median_ms must be > 0 and sigma >= 0")
        self.median_ms = float(median_ms)
        self.sigma = float(sigma)

    def sample(self, rng) -> float:
        return float(rng.lognormal(mean=math.log(self.median_ms), sigma=self.sigma))
