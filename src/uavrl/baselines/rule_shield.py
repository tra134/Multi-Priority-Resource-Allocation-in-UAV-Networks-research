"""Rule-Shield baseline: queue-triggered safety mask, no LLM (Phase 6).

If the Emergency queue exceeds theta_queue for m consecutive slots, install a
mask (Emergency bandwidth floor 0.5 + emergency power floor) for TTL slots.
A cooldown prevents re-triggering too quickly.  Variants: zero-latency and
delayed (applies the same 350 ms latency as the LLM arbiter) so Rule-Shield
can be compared fairly against the LLM arbiter.

theta_queue is tuned on validation scenarios only (never on test scenarios).
"""

from __future__ import annotations

import math

import numpy as np

from ..control.mask import Mask, project_bw_power
from .static_policies import strict_priority


class RuleShield:
    def __init__(
        self,
        theta_queue_bits,
        m=3,
        ttl_slots=600,
        cooldown=50,
        latency_slots=0,
        base_policy=strict_priority,
        bw_floor_emg=0.5,
        power_floor_emg=0.5,
        latency_model=None,
        slot_ms=5.0,
        rng=None,
    ):
        self.theta_queue_bits = float(theta_queue_bits)
        self.m = int(m)
        self.ttl_slots = int(ttl_slots)
        self.cooldown = int(cooldown)
        self.latency_slots = int(latency_slots)  # delayed mask application
        self.base_policy = base_policy
        self.bw_floor_emg = float(bw_floor_emg)
        self.power_floor_emg = float(power_floor_emg)
        # With a latency model the mask arrives after a sampled delay, drawn
        # from the same fitted distribution as the LLM arbiter (F14).
        self.latency_model = latency_model
        self.slot_ms = float(slot_ms)
        self.rng = np.random.default_rng() if rng is None else rng

        # state
        self.consec = 0
        self.cooldown_counter = 0
        self.active_mask = None          # Mask or None
        self.pending_mask = None         # (mask, applies_at_slot) for latency
        self.slot = -1
        self.trigger_count = 0
        self.mask_active_slots = 0

    def reset(self):
        self.consec = 0
        self.cooldown_counter = 0
        self.active_mask = None
        self.pending_mask = None
        self.slot = -1
        self.trigger_count = 0
        self.mask_active_slots = 0

    def __call__(self, obs, info):
        self.slot += 1
        info = info or {}

        # expire pending (latent) masks
        if self.pending_mask is not None and self.slot >= self.pending_mask[1]:
            self.active_mask = self.pending_mask[0]
            self.pending_mask = None

        # expire active mask
        if self.active_mask is not None and self.active_mask.expired(self.slot):
            self.active_mask = None

        # cooldown countdown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        q_emg = float(info.get("queue_bits", np.array([0.0, 0.0, 0.0]))[0])

        # trigger: Emergency queue above threshold for m consecutive slots
        if q_emg > self.theta_queue_bits:
            self.consec += 1
        else:
            self.consec = 0

        if (
            self.consec >= self.m
            and self.cooldown_counter == 0
            and self.active_mask is None
            and self.pending_mask is None
        ):
            bw_floor = np.array(
                [self.bw_floor_emg, 0.0, 0.0], dtype=float
            )
            if self.latency_model is not None:
                lat_slots = math.ceil(self.latency_model.sample(self.rng) / self.slot_ms)
            else:
                lat_slots = self.latency_slots
            applies_at = self.slot + lat_slots
            # TTL counts from the moment the mask is applied (as for the LLM).
            new_mask = Mask(
                bw_floor=bw_floor,
                power_floor_emg=self.power_floor_emg,
                expires_at=applies_at + self.ttl_slots,
            )
            if lat_slots > 0:
                self.pending_mask = (new_mask, applies_at)
            else:
                self.active_mask = new_mask
            self.consec = 0
            self.cooldown_counter = self.cooldown
            self.trigger_count += 1

        # base action + projection through active mask.
        # base_action is in [-1,1]^6; env maps via softmax(3*x) to shares.
        base_action = self.base_policy(obs, info)
        b_share = self._shares(base_action[:3])
        p_share = self._shares(base_action[3:])
        b, p = project_bw_power(b_share, p_share, self.active_mask)
        if self.active_mask is not None:
            self.mask_active_slots += 1
        # convert projected shares back to action space so the env's softmax
        # recovers exactly b and p.
        return np.concatenate([self._action(b), self._action(p)]).astype(np.float32)

    @staticmethod
    def _shares(x):
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def _action(shares):
        shares = np.clip(np.asarray(shares, dtype=float), 1e-9, 1.0)
        log_s = np.log(shares)
        return (log_s - np.mean(log_s)) / 3.0


class FixedReservation:
    """M13: a constant Emergency floor on top of a base policy, no trigger.

    Tells whether adaptivity is needed at all: if a permanent reservation does
    as well as the LLM, the trigger and the LLM add nothing.
    """

    def __init__(self, base_policy, bw_floor_emg=0.5, power_floor_emg=0.5):
        self.base_policy = base_policy
        self.mask = Mask(
            bw_floor=np.array([bw_floor_emg, 0.0, 0.0], dtype=float),
            power_floor_emg=float(power_floor_emg),
            expires_at=10**12,
        )
        self.mask_active_slots = 0
        self.trigger_count = 0

    def reset(self):
        self.mask_active_slots = 0

    def __call__(self, obs, info):
        base_action = self.base_policy(obs, info)
        b, p = project_bw_power(
            RuleShield._shares(base_action[:3]),
            RuleShield._shares(base_action[3:]),
            self.mask,
        )
        self.mask_active_slots += 1
        return np.concatenate(
            [RuleShield._action(b), RuleShield._action(p)]
        ).astype(np.float32)


def rule_shield(obs, info):
    """Zero-latency Rule-Shield with default tuning (theta_queue calibrated)."""
    # theta_queue tuned so shield activates under surge/shadow but not nominal.
    shield = _rule_shield_default()
    return shield(obs, info)


# Module-level default instance (stateful!).  Prefer RuleShield(...) directly
# when you need a fresh instance per evaluation.
_default_shield = None


def _rule_shield_default():
    global _default_shield
    if _default_shield is None:
        # theta_queue = ~2 slots of Emergency drain at nominal rate (~2 Mbps)
        # => threshold around 2 slots * 2e6 bps * 0.005 s = 20000 bits
        _default_shield = RuleShield(
            theta_queue_bits=20000.0, m=3, ttl_slots=600, cooldown=50,
            latency_slots=0,
        )
    return _default_shield
