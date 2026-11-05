"""Metric definitions (plan.md Section 11, Stage 1; findings F8, F9, F10)."""

import math

import numpy as np
import pandas as pd

from uavrl.eval.metrics import (
    RECOVERY_SLOTS,
    add_throughput_loss,
    episode_metrics,
    ood_events,
    trigger_quality,
)


def _steps(n=1000, event=(300, 500), violations=(), calls=(), served_imp=1000.0):
    steps = []
    for t in range(n):
        steps.append({
            "slot": t,
            "emergency_violation": t in violations,
            "ood_flag": event is not None and event[0] <= t < event[1],
            "llm_call": t in calls,
            "delay_emg_s": 0.02 if t in violations else 0.001,
            "served_emg_bits": 100.0, "served_imp_bits": served_imp,
            "served_norm_bits": 500.0,
            "arrival_emg_bits": 100.0, "drops_emg_bits": 0.0,
        })
    return steps


def test_events_found():
    flags = np.zeros(100, dtype=bool)
    flags[10:20] = True
    flags[50:55] = True
    assert ood_events(flags) == [(10, 20), (50, 55)]
    assert ood_events(np.zeros(5, dtype=bool)) == []


def test_window_violation_rates():
    viol = set(range(300, 350)) | set(range(500, 520)) | {10}
    m = episode_metrics(_steps(violations=viol))
    assert math.isclose(m["sos_violation_ood"], 50 / 200)
    assert math.isclose(m["sos_violation_recovery"], 20 / RECOVERY_SLOTS)
    assert math.isclose(m["sos_violation_episode"], 71 / 1000)


def test_delivered_bits_are_served_not_allocated():
    m = episode_metrics(_steps(served_imp=1000.0), slot_ms=5.0)
    # 1000 bits per 5 ms slot = 0.2 Mbps
    assert math.isclose(m["delivered_imp_mbps"], 0.2)


def test_loss_ratio():
    steps = _steps()
    for s in steps[:100]:
        s["drops_emg_bits"] = 10.0
    m = episode_metrics(steps)
    assert math.isclose(m["loss_ratio_emg"], 1000.0 / 100000.0)


def test_random_calls_before_event_do_not_look_like_early_detection():
    """F9: a call in the pre-event window used to give a large positive lead time."""
    m = trigger_quality(_steps(calls={250, 320}, violations={310}))
    assert math.isclose(m["detection_delay_slots"], 20)       # 320 - 300
    assert math.isclose(m["false_triggers_per_episode"], 1)    # the call at 250
    assert math.isclose(m["trigger_precision"], 0.5)
    assert math.isclose(m["trigger_recall"], 1.0)


def test_warning_margin_sign():
    early = trigger_quality(_steps(calls={305}, violations={330}))
    late = trigger_quality(_steps(calls={340}, violations={330}))
    assert early["warning_margin_slots"] == 25      # warned before damage
    assert late["warning_margin_slots"] == -10      # damage came first


def test_missed_event_and_no_event_cases():
    missed = trigger_quality(_steps(calls={700}))
    assert missed["trigger_recall"] == 0.0 and math.isnan(missed["detection_delay_slots"])
    assert missed["false_triggers_per_episode"] == 1
    nominal = trigger_quality(_steps(event=None, calls={10, 20}))
    assert nominal["false_triggers_per_episode"] == 2
    assert math.isnan(nominal["trigger_precision"] ) is False  # 0 of 2 are true positives
    assert nominal["trigger_precision"] == 0.0


def test_calls_in_recovery_window_are_not_false():
    m = trigger_quality(_steps(calls={300, 600}))
    assert m["false_triggers_per_episode"] == 0


def test_no_call_column_gives_nan():
    steps = _steps()
    for s in steps:
        s.pop("llm_call")
    assert math.isnan(trigger_quality(steps)["trigger_precision"])


def test_throughput_loss_relative_to_reference():
    rows = []
    for method, imp in [("m4_redq_sac", 10.0), ("m12_proposed", 9.0)]:
        rows.append({"method": method, "scenario": "S1", "seed": 0, "episode": 0,
                     "delivered_imp_ood_mbps": imp, "delivered_norm_ood_mbps": 10.0,
                     "delivered_imp_mbps": 0.0, "delivered_norm_mbps": 0.0})
    out = add_throughput_loss(pd.DataFrame(rows)).set_index("method")
    assert math.isclose(out.loc["m4_redq_sac", "throughput_loss_pct"], 0.0)
    assert math.isclose(out.loc["m12_proposed", "throughput_loss_pct"], 5.0)


def test_budget_violations_are_counted():
    steps = _steps(n=50, event=None)
    assert episode_metrics(steps)["budget_violation_count"] == 0
    steps[3]["b_sum"] = 1.2
    steps[7]["p_sum"] = 1.0001
    assert episode_metrics(steps)["budget_violation_count"] == 2
