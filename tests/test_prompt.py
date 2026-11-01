"""Tests for the LLM prompt builder and rate-limit helpers (Phase 8)."""

from __future__ import annotations

from uavrl.llm.client import (
    _is_rate_limit_error,
    _retry_after_seconds,
    _strip_json_fences,
)
from uavrl.llm.prompt import FEW_SHOT, build_prompt


def _features():
    return {
        "slot": 600,
        "trigger_reason": "queue",
        "U_t": 0.9677471780157282,
        "csi_emergency_mean": 0.020091172890374362,
        "csi_emergency_var": 0.004290931844828499,
        "csi_emergency_slope": 0.013108103752817669,
        "csi_important_mean": 0.727359309095329,
        "csi_important_var": 0.005540977507963289,
        "csi_important_slope": -0.018897635470277264,
        "csi_normal_mean": 1.2535131086748066,
        "csi_normal_var": 0.005427618800870855,
        "csi_normal_slope": -0.006810731340036314,
        "queue_emg_bits": 711821.6247002567,
        "queue_imp_bits": 1598014.5365139681,
        "queue_norm_bits": 676070.1756541255,
        "dQ_emg": 42569.554448890296,
        # Deliberately supplied: must never appear in the prompt.
        "ood_flag": True,
        "scenario": "surge",
    }


def test_prompt_never_contains_ground_truth_labels():
    text = build_prompt(_features())
    assert "ood_flag" not in text
    assert "scenario" not in text
    assert "True" not in text


def test_prompt_rounds_numbers_for_cache_stability():
    text = build_prompt(_features())
    assert "0.02" in text          # rounded CSI mean
    assert "0.020091" not in text
    assert "710000.0" in text or "710000" in text
    assert "967747.178" not in text


def test_few_shot_floors_are_schema_valid():
    for ex in FEW_SHOT:
        floors = ex["response"]["min_bandwidth_ratio"]
        assert sum(floors.values()) <= 0.95 + 1e-9


def test_strip_fences():
    assert _strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


class FakeRateLimit(Exception):
    status_code = 429


def test_rate_limit_helpers():
    assert _is_rate_limit_error(FakeRateLimit("rate limit"))
    assert not _is_rate_limit_error(ValueError("other"))

    class WithMessage(Exception):
        pass

    assert abs(_retry_after_seconds(WithMessage(
        "Rate limit reached ... Please try again in 5.8575s. Need more tokens?"
    )) - 5.8575) < 1e-9
    assert _retry_after_seconds(ValueError("no hint")) == 5.0
