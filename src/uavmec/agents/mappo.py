"""MAPPO learner (the paper's POS_MAPPO_NOMA baseline; spec Section 10, assumption A-23).

Per-agent stochastic actors on local observations, one centralized value network on the
global state predicting the shared return (global reward = mean of the agents' rewards).
Target and subcarrier are categorical; power_frac and bw_frac are sigmoid-squashed Gaussians.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from uavmec.agents.common import ACTION_DIM, decode_actions
from uavmec.env.types import SensorAction

_MASK = -1e9
_EPS = 1e-6


def _mlp(inp: int, hidden, out: int, act=nn.ReLU6) -> nn.Sequential:
    """MLP with the given hidden sizes (int or list), ReLU6 by default."""
    sizes = [inp, *([hidden] if isinstance(hidden, int) else list(hidden))]
    layers: list[nn.Module] = []
    for a, b in itertools.pairwise(sizes):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers, nn.Linear(sizes[-1], out))


class _PolicyNet(nn.Module):
    """Actor: logits for target (K+1) and subcarrier (M), Gaussian mean for the 2 continuous heads."""

    def __init__(self, obs_dim: int, k: int, m: int, hidden, fos: bool):
        super().__init__()
        self.k1, self.m, self.fos = k + 1, m, fos
        self.net = _mlp(obs_dim, hidden, k + 1 + m + 2)
        self.log_std = nn.Parameter(torch.zeros(2))  # A-23: state-independent std, init 1

    def dist(self, obs: torch.Tensor):
        out = self.net(obs)
        tl, sl, mu = out[..., : self.k1], out[..., self.k1 : self.k1 + self.m], out[..., -2:]
        if self.fos:  # local computing is never allowed in FOS
            tl = torch.cat([torch.full_like(tl[..., :1], _MASK), tl[..., 1:]], dim=-1)
        return torch.log_softmax(tl, -1), torch.log_softmax(sl, -1), mu

    def evaluate(self, obs, tgt, sub, u):
        """Log-prob (summed over the four heads) and entropy of the given actions."""
        tl, sl, mu = self.dist(obs)
        std = self.log_std.exp()
        x = torch.sigmoid(u)
        lp_c = (-0.5 * ((u - mu) / std) ** 2 - self.log_std - 0.9189385).sum(-1)
        lp_c = lp_c - torch.log(x * (1 - x) + _EPS).sum(-1)
        logp = tl.gather(-1, tgt[..., None])[..., 0] + sl.gather(-1, sub[..., None])[..., 0] + lp_c
        ent = -(tl.exp() * tl).sum(-1) - (sl.exp() * sl).sum(-1) + (self.log_std + 1.4189385).sum()
        return logp, ent


class MAPPO:
    """Multi-agent PPO with centralized critic. See module docstring and spec Section 10."""

    name = "MAPPO"

    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, K: int, M: int,
                 cfg: dict, fos: bool, seed: int):
        self.n, self.k, self.m, self.fos, self.cfg = n_agents, K, M, fos, dict(cfg)
        self.gen = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.actors = nn.ModuleList(
                _PolicyNet(obs_dim, K, M, cfg["actor_hidden"], fos) for _ in range(n_agents))
            self.critic = _mlp(state_dim, cfg["critic_hidden"], 1)
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.rollout: list[dict] = []
        self.episodes_in_rollout = 0
        self.n_updates = 0
        self._pending: dict | None = None

    # ---------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs: list[np.ndarray], state: np.ndarray, explore: bool) -> list[SensorAction]:
        """Sample (explore) or take the mode (greedy) action per agent; caches it for observe."""
        o = torch.as_tensor(np.stack(obs), dtype=torch.float32)
        tgt, sub, u = [], [], []
        for i, actor in enumerate(self.actors):
            tl, sl, mu = actor.dist(o[i])
            if explore:
                t = torch.multinomial(tl.exp(), 1, generator=self.gen)[0]
                s = torch.multinomial(sl.exp(), 1, generator=self.gen)[0]
                c = mu + actor.log_std.exp() * torch.randn(2, generator=self.gen)
            else:
                t, s, c = tl.argmax(), sl.argmax(), mu
            tgt.append(t), sub.append(s), u.append(c)
        tgt, sub, u = torch.stack(tgt), torch.stack(sub), torch.stack(u)
        logp = torch.stack([a.evaluate(o[i], tgt[i], sub[i], u[i])[0] for i, a in enumerate(self.actors)])
        raw = np.zeros((self.n, ACTION_DIM(self.k, self.m)), dtype=np.float32)
        raw[np.arange(self.n), tgt.numpy()] = 1.0
        raw[np.arange(self.n), self.k + 1 + sub.numpy()] = 1.0
        raw[:, -2:] = torch.sigmoid(u).numpy()
        self._pending = {"obs": o.numpy(), "tgt": tgt.numpy(), "sub": sub.numpy(),
                         "u": u.numpy(), "logp": logp.numpy()}
        return decode_actions(raw, self.k, self.m, self.fos)

    def observe(self, obs, state, rewards: np.ndarray, next_obs, next_state, done: bool) -> None:
        """Store the transition. `done` is a time-limit truncation: the next state is bootstrapped."""
        assert self._pending is not None, "observe() called before act()"
        self.rollout.append({**self._pending, "state": np.asarray(state, np.float32),
                             "next_state": np.asarray(next_state, np.float32),
                             "reward": float(np.mean(rewards)), "done": bool(done)})
        self._pending = None
        self.episodes_in_rollout += int(done)

    # -------------------------------------------------------------- learning
    def update(self) -> dict[str, float]:
        """Train once the rollout holds `rollout_episodes` episodes; otherwise return {}."""
        c = self.cfg
        if self.episodes_in_rollout < c["rollout_episodes"]:
            return {}
        f = lambda key, dt=torch.float32: torch.as_tensor(np.stack([s[key] for s in self.rollout]), dtype=dt)
        obs, state, nxt, u = f("obs"), f("state"), f("next_state"), f("u")
        tgt, sub, old_lp = f("tgt", torch.int64), f("sub", torch.int64), f("logp")
        rew, done = f("reward"), f("done")
        with torch.no_grad():
            v, nv = self.critic(state)[:, 0], self.critic(nxt)[:, 0]
        delta, adv, last = rew + c["gamma"] * nv - v, torch.zeros_like(rew), 0.0
        for t in reversed(range(len(rew))):  # GAE; episode ends cut the trace, not the bootstrap
            last = delta[t] + c["gamma"] * c["gae_lambda"] * (1 - done[t]) * last
            adv[t] = last
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        n_mb = 0
        for _ in range(c["epochs"]):
            perm = torch.randperm(len(rew), generator=self.gen)
            for idx in perm.split(c["minibatch"]):
                a_loss, ent = 0.0, 0.0
                for i, actor in enumerate(self.actors):
                    lp, e = actor.evaluate(obs[idx, i], tgt[idx, i], sub[idx, i], u[idx, i])
                    ratio = (lp - old_lp[idx, i]).exp()
                    surr = torch.min(ratio * adv[idx], ratio.clamp(1 - c["clip"], 1 + c["clip"]) * adv[idx])
                    a_loss = a_loss - surr.mean() - c["ent_coef"] * e.mean()
                    ent = ent + e.mean().item()
                self.actor_opt.zero_grad()
                (a_loss / self.n).backward()
                self.actor_opt.step()
                cl = ((self.critic(state[idx])[:, 0] - ret[idx]) ** 2).mean()
                self.critic_opt.zero_grad()
                cl.backward()
                self.critic_opt.step()
                stats["actor_loss"] += a_loss.item() / self.n
                stats["critic_loss"] += cl.item()
                stats["entropy"] += ent / self.n
                n_mb += 1
        self.rollout, self.episodes_in_rollout = [], 0
        self.n_updates += 1
        return {k: x / n_mb for k, x in stats.items()}

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Everything needed to resume exactly: networks, optimizers, rollout, counters, RNG."""
        return {"actors": self.actors.state_dict(), "critic": self.critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "rollout": [dict(s) for s in self.rollout], "pending": self._pending,
                "episodes_in_rollout": self.episodes_in_rollout, "n_updates": self.n_updates,
                "gen": self.gen.get_state()}

    def load_state_dict(self, d: dict) -> None:
        """Restore a checkpoint produced by `state_dict`."""
        self.actors.load_state_dict(d["actors"])
        self.critic.load_state_dict(d["critic"])
        self.actor_opt.load_state_dict(d["actor_opt"])
        self.critic_opt.load_state_dict(d["critic_opt"])
        self.rollout = [dict(s) for s in d["rollout"]]
        self._pending = d["pending"]
        self.episodes_in_rollout, self.n_updates = d["episodes_in_rollout"], d["n_updates"]
        self.gen.set_state(d["gen"])
