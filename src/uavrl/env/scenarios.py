"""Scenario definitions.

A scenario modifies the environment at a start slot for a duration:
traffic surge (rate multiplier), deep shadowing (extra dB), and combinations.
Scenario configuration comes from configs/scenarios.yaml (see make_scenario).  The
``info["ood_flag"]`` produced by the environment is the ground-truth OOD label
(metrics only -- never in the observation or the LLM prompt).
"""

from __future__ import annotations

import numpy as np


class Scenario:
    """A time-bounded disturbance applied to the environment.

    Parameters
    ----------
    name : str
    disturbance : dict
        Keys: type in {"surge","shadow","compound","none"},
        duration_slots, start_slot (int or [lo, hi] range), plus type-specific
        keys (surge_multiplier, extra_db/shadow_db, latency_median_ms).
    ood : bool
        Ground-truth OOD label (metrics only).
    """

    def __init__(self, name, disturbance=None, ood=False, rng=None):
        self.name = name
        self.disturbance = disturbance or {}
        self.ood = bool(ood)
        self.rng = np.random.default_rng() if rng is None else rng
        self.active = False
        self.start_slot = None
        self.end_slot = None
        self._schedule()

    def _schedule(self):
        if self.disturbance.get("type", "none") == "none":
            self.start_slot = None
            self.end_slot = None
            return
        start = self.disturbance.get("start_slot", 0)
        if isinstance(start, (list, tuple)):
            self.start_slot = int(self.rng.integers(start[0], start[1] + 1))
        else:
            self.start_slot = int(start)
        self.end_slot = self.start_slot + int(self.disturbance.get("duration_slots", 0))

    def is_active(self, slot):
        if self.start_slot is None or self.end_slot is None:
            return False
        self.active = self.start_slot <= slot < self.end_slot
        return self.active

    def multipliers(self, nominal_multipliers):
        """Per-class arrival-rate multipliers for this slot."""
        if not self.active:
            return list(nominal_multipliers)
        mult = list(nominal_multipliers)
        d = self.disturbance
        if d.get("type") in ("surge", "compound"):
            sm = float(d.get("surge_multiplier", 1.0))
            # Surge applies to the Emergency class (index 0).
            mult[0] *= sm
        return mult

    def shadow_db(self):
        """Extra shadowing offset [dB] for this slot (0.0 if inactive)."""
        if not self.active:
            return 0.0
        return float(self.disturbance.get("shadow_db", self.disturbance.get("extra_db", 0.0)))

    def latency_median_ms(self):
        """LLM latency median [s] under this scenario (None if not stressing)."""
        if not self.active:
            return None
        v = self.disturbance.get("latency_median_ms", None)
        return None if v is None else float(v)


# --------------------------------------------------------------------------- registry
def load_scenario_specs(path=None):
    """Read configs/scenarios.yaml -> (specs, aliases)."""
    from pathlib import Path

    import yaml

    from ..config import find_project_root

    path = Path(path) if path else find_project_root() / "configs" / "scenarios.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data["scenarios"], data.get("aliases", {})


def resolve_scenario_id(key, specs, aliases):
    """Accept an id (S1), a name (surge_stress) or an alias (compound)."""
    if key in specs:
        return key
    if key in aliases:
        return aliases[key]
    for sid, spec in specs.items():
        if spec.get("name") == key:
            return sid
    raise KeyError(f"unknown scenario {key!r}; known: {sorted(specs)}")


def scenario_spec(key, path=None):
    specs, aliases = load_scenario_specs(path)
    sid = resolve_scenario_id(key, specs, aliases)
    return sid, specs[sid]


def make_scenario(key, seed_or_rng=None, path=None):
    """Build a Scenario from the registry.  ``seed_or_rng`` seeds the start slot."""
    sid, spec = scenario_spec(key, path)
    if isinstance(seed_or_rng, np.random.Generator):
        rng = seed_or_rng
    else:
        rng = np.random.default_rng(seed_or_rng)
    return Scenario(spec["name"], dict(spec["disturbance"]), ood=bool(spec["ood"]), rng=rng)
