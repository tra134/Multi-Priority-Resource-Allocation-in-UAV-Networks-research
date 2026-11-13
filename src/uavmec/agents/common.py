"""Shared actor and action decoding for the learners (docs/lin2025_spec.md Section 10).

Paper: Lin et al. 2025, Section 5.3 (actor with heads of sizes (K+1, M, 2), Gumbel-Softmax
for the discrete heads). Raw action per agent = [target one-hot (K+1) | subcarrier
one-hot (M) | power_frac, bw_frac].
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from uavmec.env.types import SensorAction


def ACTION_DIM(K: int, M: int) -> int:
    """Length of one agent's raw action vector."""
    return K + 1 + M + 2


def decode_actions(raw: np.ndarray, n_uavs: int, n_sub: int, fos: bool) -> list[SensorAction]:
    """Turn raw actions (N, ACTION_DIM) into SensorActions. FOS never picks local (target 0)."""
    raw = np.asarray(raw, dtype=np.float64)
    out = []
    for row in raw:
        scores = row[: n_uavs + 1].copy()
        if fos:
            scores[0] = -np.inf
        sub = int(np.argmax(row[n_uavs + 1 : n_uavs + 1 + n_sub]))
        p, b = np.clip(row[-2:], 0.0, 1.0)
        out.append(SensorAction(int(np.argmax(scores)), sub, float(p), float(b)))
    return out


class Actor(nn.Module):
    """Local actor: 128-unit ReLU6 trunk (P: Sec. 5.3), two Gumbel heads, one sigmoid head."""

    def __init__(self, obs_dim: int, K: int, M: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU6())
        self.target_head = nn.Linear(hidden, K + 1)
        self.sub_head = nn.Linear(hidden, M)
        self.cont_head = nn.Linear(hidden, 2)

    def forward(self, obs: torch.Tensor, explore: bool, tau: float, fos: bool) -> torch.Tensor:
        """Raw action (..., ACTION_DIM). explore: hard Gumbel-Softmax sample (straight-through
        gradient); otherwise argmax one-hot. fos: local logit masked with -inf."""
        h = self.trunk(obs)
        tgt = self.target_head(h)
        if fos:
            tgt = torch.cat([torch.full_like(tgt[..., :1], float("-inf")), tgt[..., 1:]], dim=-1)
        parts = []
        for logits in (tgt, self.sub_head(h)):
            if explore:
                parts.append(F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1))
            else:
                parts.append(F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype))
        return torch.cat([*parts, torch.sigmoid(self.cont_head(h))], dim=-1)
