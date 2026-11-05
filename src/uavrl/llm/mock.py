"""Mock LLM arbiter with rule-based diagnosis and failure injection (Phase 7).

Implements the same interface as the real client.  Returns a rule-based JSON
from the prompt features (e.g. high dQ_emg + stable CSI -> SOS_SURGE with
emergency floor 0.6).  Supports failure injection: invalid JSON, out-of-range
floors, wrong diagnosis, timeout, plus a lognormal latency distribution.

The prompt is accepted (for interface parity) but the diagnosis is derived
from the structured ``features`` dict so the mock is deterministic given a seed.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

from .schema import LLMResponse, LLMInvalid


def _lognormal_ms(median_ms, sigma, rng):
    """Lognormal latency [ms] with given median and log-sigma."""
    mu = math.log(median_ms)
    return float(rng.lognormal(mean=mu, sigma=sigma))


class MockLLM:
    def __init__(
        self,
        model="mock",
        latency_median_ms=350.0,
        latency_sigma=0.6,
        failure_injection=None,
        rng=None,
    ):
        self.model = model
        self.latency_median_ms = latency_median_ms
        self.latency_sigma = latency_sigma
        self.failure_injection = dict(failure_injection or {})
        self.rng = np.random.default_rng() if rng is None else rng
        self.call_count = 0

    def call(self, prompt: str, features: Optional[dict] = None):
        """Returns (response_dict_or_None, latency_ms, ok, reason).

        ok=False means the response should be treated as invalid (fallback).
        """
        self.call_count += 1
        features = features or {}
        latency_ms = _lognormal_ms(
            self.latency_median_ms, self.latency_sigma, self.rng
        )

        fi = self.failure_injection
        rng = self.rng

        # -------- timeout failure --------
        if rng.random() < fi.get("p_timeout", 0.0):
            return None, latency_ms, False, "timeout"

        # -------- rule-based diagnosis --------
        mode, floors, power_boost = self._diagnose(features)
        ttl = int(self.rng.integers(100, 601))

        response = {
            "operating_mode": mode,
            "min_bandwidth_ratio": {
                "emergency": float(floors[0]),
                "important": float(floors[1]),
                "normal": float(floors[2]),
            },
            "power_boost_emergency": bool(power_boost),
            "ttl_duration_slots": int(ttl),
        }

        # -------- wrong diagnosis --------
        if rng.random() < fi.get("p_wrong_diagnosis", 0.0):
            alternatives = ["NORMAL", "SOS_SURGE", "DEEP_FADING", "COMPOUND"]
            if mode in alternatives:
                alternatives.remove(mode)
            response["operating_mode"] = self.rng.choice(alternatives)

        # -------- out-of-range floors --------
        if rng.random() < fi.get("p_out_of_range", 0.0):
            response["min_bandwidth_ratio"] = {
                "emergency": float(rng.uniform(0.9, 1.0)),
                "important": float(rng.uniform(0.0, 1.0)),
                "normal": float(rng.uniform(0.0, 1.0)),
            }
            response["ttl_duration_slots"] = int(rng.integers(1001, 2000))

        # -------- invalid JSON --------
        if rng.random() < fi.get("p_invalid_json", 0.0):
            return "{ not valid json }", latency_ms, False, "invalid_json"

        return response, latency_ms, True, "ok"

    # Thresholds mirror the diagnosis guide in llm/prompt.py.  The mock reads
    # only features the real prompt contains (F3); it never sees ground truth.
    CSI_FADING = 0.1            # normalized CSI mean below this = deep fading
    QUEUE_ACTIVE_BITS = 1.0e4   # Emergency queue large enough to matter
    STRONG_GROWTH_BITS = 2.0e4  # dQ_emg per slot that signals a surge on top of fading

    def _diagnose(self, features):
        dQ = float(features.get("dQ_emg", 0.0))
        q_emg = float(features.get("queue_emg_bits", 0.0))
        csi = min(
            float(features.get(f"csi_{cls}_mean", 1.0))
            for cls in ("emergency", "important", "normal")
        )

        if csi < self.CSI_FADING:
            if dQ > self.STRONG_GROWTH_BITS:
                return "COMPOUND", np.array([0.65, 0.18, 0.12]), True
            return "DEEP_FADING", np.array([0.6, 0.2, 0.15]), True
        if dQ > 0.0 and q_emg > self.QUEUE_ACTIVE_BITS:
            return "SOS_SURGE", np.array([0.6, 0.2, 0.15]), True
        return "NORMAL", np.array([0.2, 0.4, 0.35]), False


def mock_response_to_text(response: dict) -> str:
    return json.dumps(response)
