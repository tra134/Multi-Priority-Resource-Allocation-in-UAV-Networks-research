"""Traffic model: fluid queues with Poisson packet arrivals per class.

Each class generates packets of a fixed size at a Poisson rate.  The arrival
process supports a rate-multiplier hook used to model traffic surges
(emergency arrivals 4x, 8x, ...).

The queue evolves (in the environment) as:

    Q_k(t+1) = max(0, Q_k - R_k * dt) + A_k

where Q_k is in bits, R_k is the achieved rate [bps], dt is the slot duration
[s], and A_k is the arrived bits in the slot.
"""

from __future__ import annotations

import numpy as np


class TrafficSource:
    """Poisson packet source for one traffic class."""

    def __init__(self, mean_mbps, packet_bytes=200, rng=None):
        """Parameters
        ----------
        mean_mbps : float
            Mean arrival rate [Mbps].
        packet_bytes : int
            Fixed packet size [bytes].
        rng : numpy.random.Generator, optional
        """
        self.mean_mbps = float(mean_mbps)
        self.packet_bytes = int(packet_bytes)
        self.rng = np.random.default_rng() if rng is None else rng
        # Poisson rate [packets/s]
        self.lambda_packets_per_s = (mean_mbps * 1e6) / (packet_bytes * 8.0)

    def arrivals_bits(self, dt_s, multiplier=1.0):
        """Arrived bits in a slot of duration dt_s.

        Parameters
        ----------
        dt_s : float
            Slot duration [s].
        multiplier : float
            Rate-multiplier hook (1.0 = nominal, 4.0 = 4x surge, ...).
        """
        lam = self.lambda_packets_per_s * multiplier * dt_s
        n = self.rng.poisson(lam)
        return int(n * self.packet_bytes * 8)


class Traffic:
    """One traffic source per class (Emergency, Important, Normal)."""

    def __init__(self, mean_mbps_per_class, packet_bytes=200, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.sources = [
            TrafficSource(m, packet_bytes, rng=rng) for m in mean_mbps_per_class
        ]

    def arrivals_bits(self, dt_s, multipliers=None):
        """Arrived bits per class in a slot.

        Parameters
        ----------
        dt_s : float
            Slot duration [s].
        multipliers : array-like of float, optional
            Per-class rate multiplier (e.g. [4.0, 1.0, 1.0] for an Emergency surge).
        """
        if multipliers is None:
            multipliers = [1.0] * len(self.sources)
        return np.array(
            [s.arrivals_bits(dt_s, m) for s, m in zip(self.sources, multipliers)],
            dtype=float,
        )

    @property
    def mean_mbps(self):
        return np.array([s.mean_mbps for s in self.sources])
