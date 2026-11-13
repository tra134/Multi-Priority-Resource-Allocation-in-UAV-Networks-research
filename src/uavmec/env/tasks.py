"""Task generation (assumptions A-5 and A-33)."""

from __future__ import annotations

import numpy as np

from uavmec.config import LinConfig


def sample_tasks(rng: np.random.Generator, n: int, cfg: LinConfig) -> tuple[np.ndarray, np.ndarray]:
    """Draw one task per sensor: sizes C_n in bits and deadlines D_n in seconds.

    A-5: C_n = C * d_n, d_n ~ Poisson(lambda), clipped to [task_min_bits, task_max_bits].
    A-33: D_n ~ U(deadline_min_s, deadline_max_s) = U(1, 50) ms, independent of C_n
    (Table 1 read literally; replaced A-6 after test T-1, docs/lin2025_spec.md Section 5).
    """
    d = rng.poisson(cfg.task_poisson_lambda, size=n)
    bits = np.clip(cfg.task_unit_bits * d, cfg.task_min_bits, cfg.task_max_bits).astype(float)
    deadline = rng.uniform(cfg.deadline_min_s, cfg.deadline_max_s, size=n)  # A-33
    return bits, deadline
