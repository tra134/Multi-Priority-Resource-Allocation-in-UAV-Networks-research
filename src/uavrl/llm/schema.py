"""Pydantic schema for the LLM arbiter response.

The LLM returns JSON that sets bandwidth floors / power priority with a TTL.
Validator rules (plan.md Section 7):
  * all fields present
  * floors in [0, 1]
  * sum(floors) <= 0.95
  * emergency <= 0.9
  * TTL in [50, 1000]
Anything else is invalid -> safety fallback.  `reward_weight_scale` removed
(decision D4).  The prompt must never contain ground-truth labels.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class MinBandwidthRatio(BaseModel):
    emergency: float = Field(ge=0.0, le=1.0)
    important: float = Field(ge=0.0, le=1.0)
    normal: float = Field(ge=0.0, le=1.0)


class LLMResponse(BaseModel):
    operating_mode: str
    min_bandwidth_ratio: MinBandwidthRatio
    power_boost_emergency: bool
    ttl_duration_slots: int

    @field_validator("operating_mode")
    @classmethod
    def _mode(cls, v):
        allowed = {"NORMAL", "SOS_SURGE", "DEEP_FADING", "COMPOUND", "UNKNOWN"}
        if v not in allowed:
            raise ValueError(f"unknown operating_mode {v!r}")
        return v

    @field_validator("ttl_duration_slots")
    @classmethod
    def _ttl(cls, v):
        if not (50 <= v <= 1000):
            raise ValueError("ttl_duration_slots out of [50, 1000]")
        return v

    @model_validator(mode="after")
    def _floors(self):
        f = self.min_bandwidth_ratio
        s = f.emergency + f.important + f.normal
        if s > 0.95 + 1e-9:
            raise ValueError("sum of bandwidth floors > 0.95")
        if f.emergency > 0.9:
            raise ValueError("emergency bandwidth floor > 0.9")
        return self

    def to_mask(self) -> dict:
        """Convert to mask parameters for the controller."""
        return {
            "bw_floor": [
                self.min_bandwidth_ratio.emergency,
                self.min_bandwidth_ratio.important,
                self.min_bandwidth_ratio.normal,
            ],
            "power_floor_emg": 0.7 if self.power_boost_emergency else 0.5,
            "ttl_slots": self.ttl_duration_slots,
        }


class LLMInvalid(Exception):
    """Raised when an LLM response fails validation / is unparseable."""
