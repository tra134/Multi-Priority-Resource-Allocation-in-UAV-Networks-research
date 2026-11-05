"""Safety mask: bandwidth floors + emergency power floor with TTL.

Projection (minimal change, F18).  With e = max(b_raw - f, 0):
    b = f + (1 - sum(f)) * e / sum(e)  -> b >= f, sum(b) = 1, requires sum(f) <= 1
    p: same construction with the Emergency power floor
An action that already respects the floors is returned unchanged (idempotent);
the earlier map b = f + (1 - sum(f)) * b_raw altered it as well.

An invalid mask (sum(f) > 1 or any floor out of [0,1]) is rejected and treated
as an invalid LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mask:
    """An active safety mask with a TTL (expires_at = slot when it lapses)."""

    bw_floor: np.ndarray          # shape (3,), per-class bandwidth floor [0,1]
    power_floor_emg: float        # emergency power floor [0,1]
    expires_at: int               # slot index when the mask expires

    def valid(self) -> bool:
        f = np.asarray(self.bw_floor, dtype=float)
        return (
            bool(np.all(f >= 0.0)) and bool(np.all(f <= 1.0))
            and bool(0.0 <= self.power_floor_emg <= 1.0)
            and bool(f.sum() <= 1.0 + 1e-9)
        )

    def expired(self, slot: int) -> bool:
        return slot >= self.expires_at


def project_bw_power(b_raw, p_raw, mask: Mask | None):
    """Project raw bandwidth/power shares through an active mask.

    Parameters
    ----------
    b_raw, p_raw : array-like shape (3,)
        Raw shares from softmax (each sums to 1).
    mask : Mask or None
        If None, returns raw shares unchanged.

    Returns
    -------
    (b, p) projected shares, each summing to 1 and respecting floors.

    Raises
    ------
    ValueError if the mask is invalid.
    """
    b_raw = np.asarray(b_raw, dtype=float)
    p_raw = np.asarray(p_raw, dtype=float)
    if mask is None:
        return b_raw.copy(), p_raw.copy()
    if not mask.valid():
        raise ValueError("invalid mask: floors out of range or sum(bw_floor) > 1")

    f_bw = np.asarray(mask.bw_floor, dtype=float)
    f_p = np.array([mask.power_floor_emg, 0.0, 0.0], dtype=float)

    return _project_one(b_raw, f_bw), _project_one(p_raw, f_p)


def _project_one(raw, floor):
    """Smallest-change feasible share: floors met, sums to 1."""
    free = 1.0 - float(floor.sum())
    excess = np.maximum(raw - floor, 0.0)
    total = float(excess.sum())
    if total < 1e-12:
        # raw is at or below every floor: nothing above the floors to scale.
        return floor + free * raw
    return floor + free * excess / total


def project_action(action, mask: Mask | None):
    """Project a 6-dim raw action (in [-1,1]) through a mask.

    Returns the projected 6-dim action (inverse softmax of projected shares) so
    it can be fed back through the same softmax(3*x) mapping, OR the projected
    shares directly.  Here we return projected shares (b, p) for the env to use.
    """
    b_raw = _softmax_inv(action[:3])
    p_raw = _softmax_inv(action[3:])
    b, p = project_bw_power(b_raw, p_raw, mask)
    return b, p


def _softmax_inv(shares):
    shares = np.clip(np.asarray(shares, dtype=float), 1e-9, 1.0)
    log_s = np.log(shares)
    return (log_s - np.mean(log_s)) / 3.0


# --------------------------------------------------------------------------- training masks
def _softmax3(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(3.0 * x - np.max(3.0 * x))
    return e / e.sum()


def project_raw_action(action, mask: Mask) -> np.ndarray:
    """Project a raw 6-vector action through a mask; return a raw 6-vector action.

    Uses the environment's own mapping (softmax(3x) -> shares) and its inverse,
    exactly like the hybrid controller, so training and deployment agree.
    """
    action = np.asarray(action, dtype=float)
    b, p = project_bw_power(_softmax3(action[:3]), _softmax3(action[3:]), mask)
    return np.concatenate([_softmax_inv(b), _softmax_inv(p)]).astype(np.float32)


class RandomMaskSampler:
    """Random masks during training (decision D5, finding F11).

    With probability ``prob`` an episode contains one mask window: floors drawn
    uniformly under the schema limits (emergency <= 0.9, sum <= 0.95), a power
    floor of 0 or 0.7, a random start slot and a TTL in ``ttl_range``.  The
    policy therefore sees masked dynamics in training, so masks at test time
    are not a new distribution shift.
    """

    def __init__(self, prob=0.3, episode_slots=2000, ttl_range=(50, 1000), rng=None):
        self.prob = float(prob)
        self.episode_slots = int(episode_slots)
        self.ttl_range = (int(ttl_range[0]), int(ttl_range[1]))
        self.rng = np.random.default_rng() if rng is None else rng
        self.mask = None
        self.start = 0

    def new_episode(self):
        self.mask = None
        if self.rng.random() >= self.prob:
            return
        e = float(self.rng.uniform(0.2, 0.9))
        rest = 0.95 - e
        imp = float(self.rng.uniform(0.0, rest))
        norm = float(self.rng.uniform(0.0, rest - imp))
        power = 0.7 if self.rng.random() < 0.5 else 0.0
        self.start = int(self.rng.integers(0, self.episode_slots))
        ttl = int(self.rng.integers(self.ttl_range[0], self.ttl_range[1] + 1))
        self.mask = Mask(
            bw_floor=np.array([e, imp, norm]), power_floor_emg=power,
            expires_at=self.start + ttl,
        )

    def is_active(self, slot: int) -> bool:
        return self.mask is not None and self.start <= slot < self.mask.expires_at

    def apply(self, slot: int, raw_action):
        """Action executed in the environment at ``slot``."""
        if not self.is_active(slot):
            return np.asarray(raw_action, dtype=np.float32)
        return project_raw_action(raw_action, self.mask)
