"""Structured prompt builder for the LLM arbiter (Phase 7-8).

The prompt contains the sliding-window (W=50) mean/variance/slope of CSI per
class, queue lengths and dQ_emg, U_t, trigger reason, current mask, and the
output schema, plus 3 few-shot examples.  It must NEVER contain ground-truth
scenario labels or the ``ood_flag`` (prompt purity).

All numeric features are rounded to 2 significant figures so that similar
situations produce identical prompts and hit the disk cache.  The scale hints
and diagnosis guide are domain knowledge, not ground-truth labels.
"""

from __future__ import annotations

import json
import math
from typing import Optional


def _round2(x):
    """Round to 2 significant figures (cache-key stability)."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return 0.0
    return round(x, 2 - int(math.floor(math.log10(abs(x)))) - 1)


FEW_SHOT = [
    {
        "situation": "Emergency queue growing fast (dQ_emg > 0), CSI near nominal",
        "response": {
            "operating_mode": "SOS_SURGE",
            "min_bandwidth_ratio": {"emergency": 0.6, "important": 0.2, "normal": 0.15},
            "power_boost_emergency": True,
            "ttl_duration_slots": 300,
        },
    },
    {
        "situation": (
            "Emergency CSI far below nominal (deep shadow/fading), Emergency queue "
            "high because the channel cannot drain it"
        ),
        "response": {
            "operating_mode": "DEEP_FADING",
            "min_bandwidth_ratio": {"emergency": 0.5, "important": 0.3, "normal": 0.15},
            "power_boost_emergency": True,
            "ttl_duration_slots": 400,
        },
    },
    {
        "situation": "Nominal operation, all queues drained, U_t low",
        "response": {
            "operating_mode": "NORMAL",
            "min_bandwidth_ratio": {"emergency": 0.2, "important": 0.4, "normal": 0.35},
            "power_boost_emergency": False,
            "ttl_duration_slots": 100,
        },
    },
]


def build_prompt(
    features: dict,
    current_mask: Optional[dict] = None,
    constraints: Optional[dict] = None,
) -> str:
    """Build the structured prompt sent to the LLM.

    features keys (all numerics rounded to 2 sig figs):
      csi_mean/var/slope per class, queue_emg/imp/norm bits, dQ_emg,
      U_t, trigger reason, slot.
    """
    lines = []
    lines.append(
        "You are an LLM arbiter for a UAV base station serving 3 traffic classes:"
    )
    lines.append("  Emergency (delay-critical), Important (video), Normal (best effort).")
    lines.append("")
    lines.append("CURRENT SITUATION (per-slot telemetry):")
    lines.append(f"  slot: {int(features.get('slot', 0))}")
    lines.append(f"  trigger_reason: {features.get('trigger_reason', 'none')}")
    lines.append(
        f"  ensemble_uncertainty U_t: {_round2(features.get('U_t', 0.0))}"
    )
    lines.append("  CSI (mean / var / slope) per class:")
    for cls in ["emergency", "important", "normal"]:
        lines.append(
            f"    {cls}: mean={_round2(features.get(f'csi_{cls}_mean', 0.0))} "
            f"var={_round2(features.get(f'csi_{cls}_var', 0.0))} "
            f"slope={_round2(features.get(f'csi_{cls}_slope', 0.0))}"
        )
    lines.append("  Queue lengths [bits]:")
    lines.append(
        f"    emergency={_round2(features.get('queue_emg_bits', 0.0))} "
        f"important={_round2(features.get('queue_imp_bits', 0.0))} "
        f"normal={_round2(features.get('queue_norm_bits', 0.0))}"
    )
    lines.append(f"  dQ_emg [bits/slot]: {_round2(features.get('dQ_emg', 0.0))}")
    if current_mask:
        lines.append(f"  active_mask: {current_mask}")
    else:
        lines.append("  active_mask: none")
    lines.append("")
    lines.append("SCALE AND DIAGNOSIS GUIDE:")
    lines.append("  CSI is a normalized channel gain: nominal is about 1.0.")
    lines.append("  CSI mean < 0.1 means deep fading/shadowing (>20 dB loss).")
    lines.append("  dQ_emg > 0 means the Emergency queue is growing this slot.")
    lines.append("  Diagnosis rules (use CSI first, then the queue trend):")
    lines.append("    - growing Emergency queue + CSI near nominal -> SOS_SURGE")
    lines.append("    - CSI far below nominal (one or more classes) -> DEEP_FADING")
    lines.append("    - both queue growth and deep fading -> COMPOUND")
    lines.append("    - low stable queues + nominal CSI -> NORMAL")
    lines.append("  Queue growth alone is not a surge if CSI is deeply faded.")
    lines.append("")
    lines.append("CONSTRAINTS:")
    lines.append("  bandwidth floors must be in [0,1] and sum <= 0.95")
    lines.append("  emergency bandwidth floor <= 0.9")
    lines.append("  TTL in [50, 1000] slots")
    lines.append("")
    lines.append("OUTPUT (JSON only, no extra text):")
    lines.append("{")
    lines.append(
        '  "operating_mode": "NORMAL|SOS_SURGE|DEEP_FADING|COMPOUND|UNKNOWN",'
    )
    lines.append(
        '  "min_bandwidth_ratio": {"emergency": 0..1, "important": 0..1, "normal": 0..1},'
    )
    lines.append('  "power_boost_emergency": true|false,')
    lines.append('  "ttl_duration_slots": 50..1000')
    lines.append("}")
    lines.append("")
    lines.append("FEW-SHOT EXAMPLES:")
    for i, ex in enumerate(FEW_SHOT, 1):
        lines.append(f"  Example {i}: situation={ex['situation']}")
        lines.append(f"    response={json.dumps(ex['response'])}")
    lines.append("")
    lines.append("Reply with the JSON object only.")
    return "\n".join(lines)
