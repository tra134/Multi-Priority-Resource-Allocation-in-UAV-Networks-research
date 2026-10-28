"""Neural networks for SAC / REDQ-SAC.

Actor: tanh-Gaussian policy mapping the 17-dim observation to a 6-dim action
in [-1, 1], with the log-probability correction for squashing.

Critics: an ensemble of N Q-networks, each a 2x256 ReLU MLP mapping
(obs, action) -> scalar Q.  REDQ target uses the min over a random subset of
M critics; the actor uses the mean Q over all N critics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


def fanin_init(t):
    """Initialize a linear layer's weights uniformly in [-1/sqrt(fanin), +]."""
    size = t.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        fan_in = size[0]
    bound = 1.0 / np.sqrt(fan_in)
    return t.data.uniform_(-bound, bound)


class Actor(nn.Module):
    """Tanh-Gaussian actor: obs -> (mean, log_std) -> squashed action in [-1, 1]."""

    def __init__(self, obs_dim, act_dim, hidden=256, log_std_min=LOG_STD_MIN,
                 log_std_max=LOG_STD_MAX):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = mlp([obs_dim, hidden, hidden], nn.ReLU)
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        # Initialize output layers small for stable early training.
        fanin_init(self.mean_layer.weight)
        fanin_init(self.log_std_layer.weight)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """Return (action, log_prob, mean).

        action is tanh-squashed to [-1, 1].  log_prob includes the squashing
        correction (SAC Eq. 20 / Appendix C of the SAC paper).
        """
        net_out = self.net(obs)
        mean = self.mean_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pi_distribution = torch.distributions.Normal(mean, std)
        if deterministic:
            pi_action = mean
        else:
            pi_action = pi_distribution.rsample()

        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Squashing correction: log pi(a|s) = log Normal(z|s) - sum log(1 - a_i^2)
            logp_pi -= torch.sum(
                2.0 * (np.log(2.0) - pi_action - F.softplus(-2.0 * pi_action)),
                axis=-1,
            )
        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi, mean

    def get_action(self, obs, deterministic=False):
        """Single numpy action for environment interaction."""
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pi, _, _ = self.forward(obs, deterministic=deterministic, with_logprob=False)
        return pi[0].cpu().numpy()


class Critic(nn.Module):
    """Single Q-network: (obs, action) -> Q, 2x256 ReLU."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, hidden, hidden, 1], nn.ReLU)
        fanin_init(self.q[-2].weight) if isinstance(self.q[-2], nn.Linear) else None
        # Initialize the final linear layer small.
        for m in self.q.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                fanin_init(m.weight)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)


class EnsembleCritic(nn.Module):
    """N Q-networks with shared forward interface.

    forward(obs, act) returns Q values of shape (N, batch).
    target_min(obs, act, M) returns min over a random subset of M critics.
    mean_q(obs, act) returns mean Q over all N critics.
    """

    def __init__(self, obs_dim, act_dim, N=10, hidden=256):
        super().__init__()
        self.N = N
        self.critics = nn.ModuleList(
            [Critic(obs_dim, act_dim, hidden) for _ in range(N)]
        )

    def forward(self, obs, act):
        qs = [critic(obs, act) for critic in self.critics]
        return torch.stack(qs, dim=0)  # (N, batch)

    def target_min(self, obs, act, M, rng=None):
        """Min Q over a random subset of M critics (REDQ target)."""
        qs = self.forward(obs, act)  # (N, batch)
        N = qs.shape[0]
        m = min(M, N)
        if rng is None:
            idx = torch.randperm(N)[:m]
        else:
            idx = torch.tensor(
                rng.choice(N, size=m, replace=False), dtype=torch.long
            )
        return qs[idx].min(dim=0).values

    def mean_q(self, obs, act):
        """Mean Q over all N critics (used for the actor loss)."""
        qs = self.forward(obs, act)  # (N, batch)
        return qs.mean(dim=0)

    def all_q(self, obs, act):
        """All N Q values (used for uncertainty / ensemble std)."""
        return self.forward(obs, act)
