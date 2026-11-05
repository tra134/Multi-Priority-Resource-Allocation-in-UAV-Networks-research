"""Episode metrics (plan.md Section 11, Stage 1; findings F8, F9, F10).

All functions take the per-slot ``steps`` list produced by
``eval.runner.run_episode`` or ``HybridController.run_episode``.

Definitions
-----------
event            a maximal run of slots with ``ood_flag`` set (ground truth).
recovery window  the RECOVERY_SLOTS slots after an event ends.
sos_violation_*  fraction of slots with Emergency delay above D_th, over the
                 whole episode, inside events only (primary), or in recovery.
delivered_*      bits actually served per second (not allocated capacity).
call             one LLM request (``llm_call`` in the step log).
true-positive call
                 a call inside an event or its recovery window.
precision_call   true-positive calls / all calls.
recall_event     events with at least one call during the event / all events.
detection_delay  first call inside the event minus the event start [slots].
warning_margin   first violation inside the event minus the first call inside
                 the event; positive means the call came before the damage.
false_triggers   calls outside every event and recovery window, per episode.
"""

from __future__ import annotations

import math

import numpy as np

RECOVERY_SLOTS = 200
NAN = float("nan")


def ood_events(ood_flags):
    """Return a list of (start, end) slot pairs, end exclusive."""
    flags = np.asarray(ood_flags, dtype=bool)
    padded = np.r_[False, flags, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else NAN


def _masks(n, events, recovery=RECOVERY_SLOTS):
    in_event = np.zeros(n, dtype=bool)
    in_recovery = np.zeros(n, dtype=bool)
    for start, end in events:
        in_event[start:end] = True
        in_recovery[end:min(n, end + recovery)] = True
    return in_event, in_recovery & ~in_event


def trigger_quality(steps, recovery=RECOVERY_SLOTS):
    """Per-call precision, per-event recall, detection delay, warning margin."""
    n = len(steps)
    empty = {
        "trigger_precision": NAN, "trigger_recall": NAN,
        "detection_delay_slots": NAN, "warning_margin_slots": NAN,
        "false_triggers_per_episode": NAN,
    }
    if n == 0 or "llm_call" not in steps[0]:
        return empty

    ood = np.array([bool(s["ood_flag"]) for s in steps])
    viol = np.array([bool(s["emergency_violation"]) for s in steps])
    calls = np.flatnonzero([bool(s["llm_call"]) for s in steps])
    events = ood_events(ood)
    in_event, in_recovery = _masks(n, events, recovery)
    covered = in_event | in_recovery

    out = dict(empty)
    out["false_triggers_per_episode"] = float(np.sum(~covered[calls])) if calls.size else 0.0
    if calls.size:
        out["trigger_precision"] = float(np.mean(covered[calls]))
    if not events:
        return out

    detected, delays, margins = 0, [], []
    for start, end in events:
        inside = calls[(calls >= start) & (calls < end)]
        if inside.size == 0:
            continue
        detected += 1
        first_call = int(inside[0])
        delays.append(first_call - start)
        v = np.flatnonzero(viol[start:end])
        if v.size:
            margins.append(int(v[0]) + start - first_call)
    out["trigger_recall"] = detected / len(events)
    out["detection_delay_slots"] = _mean(delays)
    out["warning_margin_slots"] = _mean(margins)
    return out


def episode_metrics(steps, slot_ms=5.0, recovery=RECOVERY_SLOTS):
    """All per-episode metrics that do not depend on another method."""
    n = len(steps)
    if n == 0:
        return {}
    dt = slot_ms / 1000.0
    viol = np.array([bool(s["emergency_violation"]) for s in steps])
    ood = np.array([bool(s["ood_flag"]) for s in steps])
    events = ood_events(ood)
    in_event, in_recovery = _masks(n, events, recovery)

    def served_mbps(key, mask=None):
        v = np.array([s[key] for s in steps], dtype=float)
        if mask is not None:
            v = v[mask]
        return float(v.sum() / (v.size * dt) / 1e6) if v.size else NAN

    delay = np.array([s["delay_emg_s"] for s in steps], dtype=float)
    arrived = float(sum(s["arrival_emg_bits"] for s in steps))
    dropped = float(sum(s["drops_emg_bits"] for s in steps))

    out = {
        "sos_violation_episode": float(viol.mean()),
        "sos_violation_ood": _mean(viol[in_event]) if in_event.any() else NAN,
        "sos_violation_recovery": _mean(viol[in_recovery]) if in_recovery.any() else NAN,
        "sos_p99_delay_s": float(np.percentile(delay, 99)),
        "sos_p99_delay_ood_s": (
            float(np.percentile(delay[in_event], 99)) if in_event.any() else NAN
        ),
        "delivered_emg_mbps": served_mbps("served_emg_bits"),
        "delivered_imp_mbps": served_mbps("served_imp_bits"),
        "delivered_norm_mbps": served_mbps("served_norm_bits"),
        "delivered_imp_ood_mbps": (
            served_mbps("served_imp_bits", in_event) if in_event.any() else NAN
        ),
        "delivered_norm_ood_mbps": (
            served_mbps("served_norm_bits", in_event) if in_event.any() else NAN
        ),
        "loss_ratio_emg": dropped / arrived if arrived > 0 else NAN,
        "total_drops_emg_bits": dropped,
        "n_events": len(events),
        # Measured: allocated shares must never exceed the B_max / P_max budgets.
        "budget_violation_count": int(sum(
            (s.get("b_sum", 1.0) > 1.0 + 1e-6) or (s.get("p_sum", 1.0) > 1.0 + 1e-6)
            for s in steps
        )),
    }
    out.update(trigger_quality(steps, recovery))
    return out


def add_throughput_loss(df, reference="m4_redq_sac"):
    """Add ``throughput_loss_pct`` relative to a reference method (slide 41 target).

    Compared per (scenario, seed, episode).  Positive means less delivered
    Important+Normal traffic than the reference during the event window
    (or the whole episode when the scenario has no event).
    """
    key = ["scenario", "seed", "episode"]
    df = df.copy()

    def total(frame):
        imp = frame["delivered_imp_ood_mbps"].where(
            frame["delivered_imp_ood_mbps"].notna(), frame["delivered_imp_mbps"])
        norm = frame["delivered_norm_ood_mbps"].where(
            frame["delivered_norm_ood_mbps"].notna(), frame["delivered_norm_mbps"])
        return imp + norm

    df["_tot"] = total(df)
    ref = df[df["method"] == reference].set_index(key)["_tot"].rename("_ref")
    df = df.join(ref, on=key)
    df["throughput_loss_pct"] = 100.0 * (1.0 - df["_tot"] / df["_ref"])
    return df.drop(columns=["_tot", "_ref"])


def finite_or_blank(value):
    """CSV helper: NaN/inf -> empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
