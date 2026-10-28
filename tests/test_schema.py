"""Tests for the LLM response schema (plan.md Phase 7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uavrl.llm.schema import LLMResponse, LLMInvalid


def valid_payload():
    return {
        "operating_mode": "SOS_SURGE",
        "min_bandwidth_ratio": {"emergency": 0.5, "important": 0.25, "normal": 0.2},
        "power_boost_emergency": True,
        "ttl_duration_slots": 300,
    }


def test_valid_response_parses():
    r = LLMResponse(**valid_payload())
    assert r.operating_mode == "SOS_SURGE"
    assert r.min_bandwidth_ratio.emergency == 0.5
    assert r.to_mask()["power_floor_emg"] == 0.7
    assert r.to_mask()["ttl_slots"] == 300


def test_invalid_mode_rejected():
    p = valid_payload()
    p["operating_mode"] = "BOGUS"
    with pytest.raises(ValidationError):
        LLMResponse(**p)


def test_sum_floors_gt_095_rejected():
    p = valid_payload()
    p["min_bandwidth_ratio"] = {"emergency": 0.4, "important": 0.4, "normal": 0.4}
    with pytest.raises(ValidationError):
        LLMResponse(**p)


def test_emergency_floor_gt_09_rejected():
    p = valid_payload()
    p["min_bandwidth_ratio"] = {"emergency": 0.95, "important": 0.0, "normal": 0.0}
    with pytest.raises(ValidationError):
        LLMResponse(**p)


def test_ttl_out_of_range_rejected():
    p = valid_payload()
    p["ttl_duration_slots"] = 30
    with pytest.raises(ValidationError):
        LLMResponse(**p)
    p["ttl_duration_slots"] = 2000
    with pytest.raises(ValidationError):
        LLMResponse(**p)


def test_normal_power_floor_05():
    p = valid_payload()
    p["power_boost_emergency"] = False
    r = LLMResponse(**p)
    assert r.to_mask()["power_floor_emg"] == 0.5
