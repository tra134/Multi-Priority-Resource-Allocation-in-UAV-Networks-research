"""MockLLM must be able to produce every mode from real controller features (F3)."""

import numpy as np

from uavrl.llm.mock import MockLLM


def _features(csi=1.0, q=2000.0, dq=-100.0):
    f = {"queue_emg_bits": q, "dQ_emg": dq}
    for cls in ("emergency", "important", "normal"):
        f[f"csi_{cls}_mean"] = csi
    return f


def _mode(features):
    llm = MockLLM(latency_median_ms=100.0, rng=np.random.default_rng(0))
    resp, _, ok, _ = llm.call("prompt", features)
    assert ok
    return resp["operating_mode"]


def test_all_modes_are_reachable():
    assert _mode(_features()) == "NORMAL"
    assert _mode(_features(q=5e4, dq=2e4)) == "SOS_SURGE"
    assert _mode(_features(csi=0.02, q=3e3, dq=100.0)) == "DEEP_FADING"
    assert _mode(_features(csi=0.02, q=1e5, dq=5e4)) == "COMPOUND"


def test_mock_ignores_ground_truth_keys():
    plain = _mode(_features())
    leaked = _features()
    leaked.update({"shadow_db": 25.0, "ood_flag": True, "scenario": "surge"})
    assert _mode(leaked) == plain
