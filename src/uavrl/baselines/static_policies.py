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


def strict_priority(obs, info):
    """All bandwidth and power to the Emergency class."""
    return shares_to_action([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])


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
    "fair_share": fair_share,
    "greedy_sum_rate": greedy_sum_rate,
}
