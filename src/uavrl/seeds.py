"""Disjoint seed ranges for training, monitoring, validation and test.

plan.md rule 7 and finding F17: evaluation seeds must never be used in
training.  Every script takes its seeds from here so the ranges cannot drift
into each other (``tests/test_seeds.py`` checks this).

    test        [10_000, 20_000)   final results only; 10_000 + train_seed * 1000 + episode
    validation  [20_000, 30_000)   tuning (tau_unc, theta_queue, scenario calibration)
    monitoring  [30_000, 40_000)   evaluations printed during training
    training    [1_000_000, ...)   one block of 1_000_000 per training seed
"""

from __future__ import annotations

TEST_BASE = 10_000
VAL_BASE = 20_000
MONITOR_BASE = 30_000
TRAIN_BASE = 1_000_000
TRAIN_BLOCK = 1_000_000

MAX_TRAIN_SEEDS = 10       # test seeds assume train_seed < 10 and episode < 1000
MAX_EPISODES = 1000


def test_seed(train_seed: int, episode: int) -> int:
    """Seed of one final-evaluation episode (run_grid.py)."""
    if not (0 <= train_seed < MAX_TRAIN_SEEDS and 0 <= episode < MAX_EPISODES):
        raise ValueError("train_seed must be < 10 and episode < 1000")
    return TEST_BASE + train_seed * MAX_EPISODES + episode


def val_seed(index: int) -> int:
    """Seed of one validation episode (tuning only)."""
    if not (0 <= index < 10_000):
        raise ValueError("validation index must be in [0, 10000)")
    return VAL_BASE + index


def monitor_seed(index: int) -> int:
    """Seed of one training-time monitoring episode."""
    if not (0 <= index < 10_000):
        raise ValueError("monitoring index must be in [0, 10000)")
    return MONITOR_BASE + index


def train_reset_seed(train_seed: int, episode_counter: int) -> int:
    """Seed passed to ``env.reset`` for training episode number ``episode_counter``."""
    if train_seed < 0 or not (0 <= episode_counter < TRAIN_BLOCK):
        raise ValueError("train_seed must be >= 0 and episode_counter < 1_000_000")
    return TRAIN_BASE + train_seed * TRAIN_BLOCK + episode_counter
