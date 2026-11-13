"""Shared data types for the Lin et al. 2025 environment (docs/lin2025_spec.md, Section 9)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorAction:
    """One sensor agent's action in one slot (spec Section 3).

    target: 0 = compute locally, k in 1..K = offload to UAV k-1.
    subcarrier: 0..M-1, ignored when target == 0.
    power_frac: transmit power as a fraction of p_max, in [0, 1].
    bw_frac: bandwidth as a fraction of the subcarrier bandwidth B_m, in [0, 1].
    """

    target: int
    subcarrier: int = 0
    power_frac: float = 1.0
    bw_frac: float = 1.0


@dataclass
class TaskOutcome:
    """What happened to one sensor's task in one slot.

    fail_reason is one of "deadline", "energy", "rate", "decode", or None on success.
    """

    size_bits: float
    deadline_s: float
    mode: str  # "local" | "offload"
    uav: int | None
    subcarrier: int | None
    sinr: float | None
    rate_bps: float | None
    decoded: bool | None
    latency_s: float
    energy_j: float
    success: bool
    fail_reason: str | None
