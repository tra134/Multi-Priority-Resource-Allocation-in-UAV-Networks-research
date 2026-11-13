"""MADDPG with CTDE (Lin et al. 2025, Section 5.2-5.3, Algorithm 1; docs/lin2025_spec.md Sec. 10).

Per-agent actor (local obs) and per-agent critic (global state + all agents' raw actions),
target networks with soft updates, one replay buffer of global transitions.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from uavmec.agents.common import ACTION_DIM, Actor, decode_actions
from uavmec.env.types import SensorAction


class MADDPG:
    name = "maddpg"

    def __init__(self, n_agents, obs_dim, state_dim, K, M, cfg: dict, fos: bool, seed: int):
        self.N, self.K, self.M, self.cfg, self.fos = n_agents, K, M, cfg, fos
        self.A = ACTION_DIM(K, M)
        self.np_rng = np.random.default_rng(seed)
        self._saved = torch.get_rng_state()  # caller's global torch RNG, restored after init
        torch.manual_seed(seed)
        h1, h2 = cfg["critic_hidden"]
        self.actors = [Actor(obs_dim, K, M, cfg["actor_hidden"]) for _ in range(n_agents)]
        self.critics = [
            nn.Sequential(nn.Linear(state_dim + n_agents * self.A, h1), nn.ReLU6(),
                          nn.Linear(h1, h2), nn.ReLU6(), nn.Linear(h2, 1))
            for _ in range(n_agents)
        ]
        self.t_actors, self.t_critics = copy.deepcopy(self.actors), copy.deepcopy(self.critics)
        self.a_opt = [torch.optim.Adam(a.parameters(), lr=cfg["actor_lr"]) for a in self.actors]
        self.c_opt = [torch.optim.Adam(c.parameters(), lr=cfg["critic_lr"]) for c in self.critics]
        self._trng = torch.get_rng_state()  # learner-owned torch RNG (swapped in around each use)
        torch.set_rng_state(self._saved)
        cap, N = cfg["buffer"], n_agents
        self.buf = {k: np.zeros((cap, *s), np.float32) for k, s in [
            ("obs", (N, obs_dim)), ("state", (state_dim,)), ("act", (N, self.A)), ("rew", (N,)),
            ("nobs", (N, obs_dim)), ("nstate", (state_dim,)), ("terminal", ())]}
        self.size, self.ptr, self.steps, self.updates = 0, 0, 0, 0
        self._last = None

    def _swap(self):
        self._saved = torch.get_rng_state()
        torch.set_rng_state(self._trng)

    def _unswap(self):
        self._trng = torch.get_rng_state()
        torch.set_rng_state(self._saved)

    def _sigma(self) -> float:  # A-23: linear decay of exploration noise
        c = self.cfg
        f = min(1.0, self.steps / max(1, c["noise_decay_steps"]))
        return c["noise_sigma_start"] + f * (c["noise_sigma_end"] - c["noise_sigma_start"])

    def act(self, obs, state, explore: bool) -> list[SensorAction]:
        self._swap()
        try:
            with torch.no_grad():
                o = torch.as_tensor(np.stack(obs), dtype=torch.float32)
                raw = torch.stack([self.actors[n](o[n], explore, self.cfg["gumbel_tau"], self.fos)
                                   for n in range(self.N)]).numpy().copy()
        finally:
            self._unswap()
        if explore:
            noise = self.np_rng.normal(0, self._sigma(), (self.N, 2))
            raw[:, -2:] = np.clip(raw[:, -2:] + noise, 0, 1)
        self._last = raw
        return decode_actions(raw, self.K, self.M, self.fos)

    def observe(self, obs, state, rewards, next_obs, next_state, done: bool) -> None:
        """Store one transition. `done` is a time-limit truncation, so terminal stays False
        (no bootstrap cut; time-limit truncation, spec Section 10)."""
        i = self.ptr
        for k, v in [("obs", np.stack(obs)), ("state", state), ("act", self._last), ("rew", rewards),
                     ("nobs", np.stack(next_obs)), ("nstate", next_state), ("terminal", 0.0)]:
            self.buf[k][i] = v
        self.ptr = (i + 1) % self.cfg["buffer"]
        self.size = min(self.size + 1, self.cfg["buffer"])
        self.steps += 1

    def update(self) -> dict[str, float]:
        """One gradient step for every agent (Algorithm 1 critic + actor update, soft targets)."""
        c = self.cfg
        if self.size < max(c["warmup"], c["batch"]):
            return {}
        idx = self.np_rng.integers(0, self.size, c["batch"])
        b = {k: torch.as_tensor(v[idx]) for k, v in self.buf.items()}
        B, N, tau = c["batch"], self.N, c["gumbel_tau"]
        c_loss = a_loss = 0.0
        self._swap()
        try:
            with torch.no_grad():  # next joint action from the target actors
                na = torch.stack([self.t_actors[n](b["nobs"][:, n], False, tau, self.fos)
                                  for n in range(N)], 1).reshape(B, -1)
                nin = torch.cat([b["nstate"], na], 1)
            joint = b["act"].reshape(B, -1)
            for n in range(N):
                with torch.no_grad():
                    nonterm = 1 - b["terminal"][:, None]
                    y = b["rew"][:, n:n + 1] + c["gamma"] * nonterm * self.t_critics[n](nin)
                q = self.critics[n](torch.cat([b["state"], joint], 1))
                loss = nn.functional.mse_loss(q, y)
                self.c_opt[n].zero_grad()
                loss.backward()
                self.c_opt[n].step()
                acts = b["act"].clone()
                acts[:, n] = self.actors[n](b["obs"][:, n], True, tau, self.fos)
                pl = -self.critics[n](torch.cat([b["state"], acts.reshape(B, -1)], 1)).mean()
                self.a_opt[n].zero_grad()
                pl.backward()
                self.a_opt[n].step()
                c_loss += loss.item() / N
                a_loss += pl.item() / N
            with torch.no_grad():  # soft update with tau_soft
                pairs = list(zip(self.actors, self.t_actors)) + list(zip(self.critics, self.t_critics))
                for net, tgt in pairs:
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - c["tau_soft"]).add_(c["tau_soft"] * p)
        finally:
            self._unswap()
        self.updates += 1
        return {"critic_loss": c_loss, "actor_loss": a_loss}

    def state_dict(self) -> dict:
        # deepcopy: torch state_dicts hold live tensors (Adam state would be shared otherwise)
        return copy.deepcopy({
            **{k: [m.state_dict() for m in getattr(self, k)]
               for k in ("actors", "critics", "t_actors", "t_critics")},
            "a_opt": [o.state_dict() for o in self.a_opt],
            "c_opt": [o.state_dict() for o in self.c_opt],
            "buf": self.buf,
            "counters": (self.size, self.ptr, self.steps, self.updates),
            "np_rng": self.np_rng.bit_generator.state,
            "torch_rng": self._trng,
        })

    def load_state_dict(self, d: dict) -> None:
        d = copy.deepcopy(d)
        for k in ("actors", "critics", "t_actors", "t_critics"):
            for m, s in zip(getattr(self, k), d[k]):
                m.load_state_dict(s)
        for o, s in zip(self.a_opt + self.c_opt, d["a_opt"] + d["c_opt"]):
            o.load_state_dict(s)
        for k, v in d["buf"].items():
            self.buf[k][:] = v
        self.size, self.ptr, self.steps, self.updates = d["counters"]
        self.np_rng.bit_generator.state = copy.deepcopy(d["np_rng"])
        self._trng = d["torch_rng"].clone()
