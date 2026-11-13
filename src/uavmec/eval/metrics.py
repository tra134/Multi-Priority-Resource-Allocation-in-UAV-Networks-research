"""Metrics for the Lin et al. 2025 reproduction (docs/lin2025_spec.md, Sections 4 and 9).

Definitions:
- success rate = successful tasks / all tasks
- mean latency = mean over successful tasks only, in ms
- mean energy = sensor energy per task, mean over successful tasks only (A-34, test T-3),
  in mW (mJ per task with one task per 1 s slot)
- counts by mode and a fail-reason histogram
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from uavmec.env.types import TaskOutcome

FAIL_REASONS = ("deadline", "energy", "rate", "decode")


def summarize_outcomes(slots: Sequence[Sequence[TaskOutcome]]) -> dict:
    """Aggregate one episode given a list (per slot) of lists of TaskOutcome."""
    flat = [o for slot in slots for o in slot]
    n = len(flat)
    ok = [o for o in flat if o.success]
    modes = Counter(o.mode for o in flat)
    fails = Counter(o.fail_reason for o in flat if not o.success)
    return {
        "n_tasks": n,
        "n_success": len(ok),
        "success_rate": len(ok) / n if n else float("nan"),
        "mean_latency_ms": 1000.0 * float(np.mean([o.latency_s for o in ok])) if ok else float("nan"),
        # A-34: averaged over successful tasks, the same set as latency (T-3)
        "mean_energy_mw": 1000.0 * float(np.mean([o.energy_j for o in ok])) if ok else float("nan"),
        "n_local": modes.get("local", 0),
        "n_offload": modes.get("offload", 0),
        **{f"fail_{r}": fails.get(r, 0) for r in FAIL_REASONS},
    }


def fail_histogram(slots: Sequence[Sequence[TaskOutcome]]) -> dict[str, int]:
    """Histogram of fail_reason over failed tasks."""
    return dict(Counter(o.fail_reason for slot in slots for o in slot if not o.success))


def aggregate_episodes(rows: Sequence[dict]) -> dict:
    """Pool per-episode summaries: ratios are recomputed from pooled counts, not averaged."""
    n = sum(r["n_tasks"] for r in rows)
    ns = sum(r["n_success"] for r in rows)
    lat_num = sum(r["mean_latency_ms"] * r["n_success"] for r in rows if r["n_success"])
    en_num = sum(r["mean_energy_mw"] * r["n_success"] for r in rows if r["n_success"])  # A-34
    out = {
        "n_episodes": len(rows),
        "n_tasks": n,
        "n_success": ns,
        "success_rate": ns / n if n else float("nan"),
        "mean_latency_ms": lat_num / ns if ns else float("nan"),
        "mean_energy_mw": en_num / ns if ns else float("nan"),
    }
    for key in ("n_local", "n_offload", *[f"fail_{r}" for r in FAIL_REASONS]):
        out[key] = sum(r[key] for r in rows)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over episodes (NaNs dropped), fixed rng."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
