"""Static (non-learning) baselines.

Each policy maps ``(obs, info) -> action`` where action is a 6-vector in
[-1, 1] that the environment turns into bandwidth/power shares via
``softmax(3 * x)``.  Helper :func:`shares_to_action` inverts the softmax so a
policy can specify exact shares.
"""

from __future__ import annotations

import numpy as np


def shares_to_action(b_share, p_share):
    """Convert bandwidth/power shares (sum to 1 each) to the raw action in [-1, 1].

    Inverse of softmax(3 * x):  x_i = (log(s_i) - mean(log(s))) / 3
    """
    b_share = np.asarray(b_share, dtype=float)
    p_share = np.asarray(p_share, dtype=float)
    eps = 1e-9

    def _inv(s):
        s = np.clip(np.asarray(s, dtype=float), eps, 1.0)
        log_s = np.log(s)
        return (log_s - np.mean(log_s)) / 3.0

    return np.concatenate([_inv(b_share), _inv(p_share)]).astype(np.float32)


def emergency_only(obs, info):
    """All bandwidth and power to Emergency, always.

    Upper bound on Emergency protection, NOT a fair baseline: it starves the
    other classes even when the Emergency queue is empty (finding F1).
    """
    return shares_to_action([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])


class QueueAwarePriority:
    """Strict priority that is work-conserving (M1, finding F1).

    Classes are served in the order Emergency > Important > Normal.  Each gets
    the share it needs to drain its queue plus an arrival allowance this slot,
    capped by what is left.  Leftover capacity is spread by nominal load, so
    nothing is wasted.  Power follows bandwidth, which keeps the per-Hz SNR
    constant, so the rate is linear in the share: c_k = B * log2(1 + P*g_k/(N0*B)).

    Uses the previous slot's queues and gains from ``info`` (one slot of CSI
    delay); on the first slot, where ``info`` is empty, it uses fixed weights.
    The Emergency margin (3.0) was tuned on validation seeds 20000+ as a
    trade-off between Emergency violations and delivered Normal traffic.
    """

    DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)

    def __init__(self, dt=0.005, B_max=1.0e7, P_max=1.0, n0_w_per_hz=None,
                 lam_bps=(2.0e6, 35.0e6, 28.0e6), margin_emg=3.0, margin_other=1.0):
        if n0_w_per_hz is None:
            n0_w_per_hz = 10.0 ** ((-174.0 + 7.0 - 30.0) / 10.0)
        self.dt = float(dt)
        self.B_max = float(B_max)
        self.P_max = float(P_max)
        self.n0 = float(n0_w_per_hz)
        self.lam = np.asarray(lam_bps, dtype=float)
        # Emergency has a 15 ms deadline, so it gets an arrival allowance; the
        # other classes tolerate delay, so over-reserving would only waste capacity.
        self.margin = np.array([margin_emg, margin_other, margin_other], dtype=float)

    @classmethod
    def from_config(cls, cfg, margin_emg=3.0, margin_other=1.0):
        t, c, tr = cfg.get("timing", {}), cfg.get("channel", {}), cfg.get("traffic", {})
        n0 = 10.0 ** ((float(c.get("noise_dbm_hz", -174.0))
                       + float(c.get("noise_figure_db", 7.0)) - 30.0) / 10.0)
        lam = [float(tr.get("emergency_mean_mbps", 2.0)),
               float(tr.get("important_mean_mbps", 35.0)),
               float(tr.get("normal_mean_mbps", 28.0))]
        return cls(dt=float(t.get("slot_ms", 5.0)) / 1000.0,
                   B_max=float(c.get("B_max", 1.0e7)), P_max=float(c.get("P_max", 1.0)),
                   n0_w_per_hz=n0, lam_bps=[x * 1e6 for x in lam],
                   margin_emg=margin_emg, margin_other=margin_other)

    def shares(self, info):
        info = info or {}
        if "queue_bits" not in info or "gain" not in info:
            return np.asarray(self.DEFAULT_WEIGHTS, dtype=float)
        q = np.asarray(info["queue_bits"], dtype=float)
        g = np.maximum(np.asarray(info["gain"], dtype=float), 1e-30)
        cap_per_share = self.B_max * np.log2(1.0 + self.P_max * g / (self.n0 * self.B_max))
        need_bps = q / self.dt + self.margin * self.lam
        s = np.zeros(3)
        left = 1.0
        for k in range(3):  # Emergency, Important, Normal
            s[k] = min(left, need_bps[k] / max(cap_per_share[k], 1e-9))
            left -= s[k]
        if left > 1e-12:
            s += left * self.lam / self.lam.sum()
        return s

    def __call__(self, obs, info):
        s = self.shares(info)
        return shares_to_action(s, s)


_DEFAULT_PRIORITY = QueueAwarePriority()


def strict_priority(obs, info):
    """Default-configured queue-aware strict priority (see QueueAwarePriority)."""
    return _DEFAULT_PRIORITY(obs, info)


def fair_share(obs, info):
    """Equal shares for all three classes."""
    return shares_to_action([1.0 / 3.0] * 3, [1.0 / 3.0] * 3)


def greedy_sum_rate(obs, info):
    """All resources to the class with the best instantaneous channel gain."""
    gain = np.asarray(info.get("gain", [1.0, 1.0, 1.0]), dtype=float)
    k = int(np.argmax(gain))
    b = [0.0, 0.0, 0.0]
    p = [0.0, 0.0, 0.0]
    b[k] = 1.0
    p[k] = 1.0
    return shares_to_action(b, p)


POLICIES = {
    "strict_priority": strict_priority,
    "emergency_only": emergency_only,
    "fair_share": fair_share,
    "greedy_sum_rate": greedy_sum_rate,
}
