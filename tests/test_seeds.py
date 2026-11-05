"""Seed ranges must stay disjoint (plan.md rule 7, finding F17)."""

import pytest

from uavrl import seeds


def _range(fn, n):
    return {fn(i) for i in range(n)}


def test_ranges_are_disjoint():
    test = {seeds.test_seed(s, e) for s in range(seeds.MAX_TRAIN_SEEDS)
            for e in range(seeds.MAX_EPISODES)}
    val = _range(seeds.val_seed, 10_000)
    mon = _range(seeds.monitor_seed, 10_000)
    assert not test & val
    assert not test & mon
    assert not val & mon
    assert max(test) < seeds.VAL_BASE <= min(val) and max(val) < seeds.MONITOR_BASE


def test_training_seeds_never_hit_evaluation_seeds():
    eval_max = seeds.MONITOR_BASE + 10_000
    for train_seed in range(5):
        first = seeds.train_reset_seed(train_seed, 0)
        last = seeds.train_reset_seed(train_seed, seeds.TRAIN_BLOCK - 1)
        assert first >= eval_max
        if train_seed < 4:
            assert last < seeds.train_reset_seed(train_seed + 1, 0)


def test_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        seeds.test_seed(0, seeds.MAX_EPISODES)
    with pytest.raises(ValueError):
        seeds.val_seed(10_000)
    with pytest.raises(ValueError):
        seeds.train_reset_seed(0, seeds.TRAIN_BLOCK)
