"""Trivial placeholder test so `pytest` passes in Phase 0."""

import numpy as np


def test_numpy_available():
    assert np.__version__ is not None


def test_placeholder_true():
    assert True
