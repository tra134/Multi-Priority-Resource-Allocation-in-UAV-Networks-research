"""SAC agent with REDQ-style ensemble critic.

Implements Soft Actor-Critic (Haarnoja et al. 2018) with:

* tanh-Gaussian actor with automatic entropy tuning (target_entropy = -act_dim)
* N-critic ensemble; target Q = min over a random subset of M=2 critics (REDQ)
* actor loss uses the mean Q over all N critics
* one gradient step per env step (G=1) by default

Based on CleanRL's sac_continuous_action.py, adapted to our ensemble-critic
design.  Runs on CPU or CUDA.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import Actor, EnsembleCritic


class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        device="cpu",
        seed=0,
        N_critics=10,
        gamma=0.99,
        lr=3e-4,
        tau=0.005,
        alpha=None,            # None -> automatic entropy tuning
        target_entropy=None,
        hidden=256,
        M=2,                   # REDQ subset size for target
        G=1,                   # update-to-data ratio
        rng=None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.M = M
        self.G = G
        self.rng = np.random.default_rng() if rng is None else rng

        # Actor + ensemble critic.
        self.actor = Actor(obs_dim, act_dim, hidden=hidden).to(self.device)
        self.critic = EnsembleCritic(obs_dim, act_dim, N=N_critics, hidden=hidden).to(self.device)
        self.critic_target = EnsembleCritic(
            obs_dim, act_dim, N=N_critics, hidden=hidden
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning.
        if target_entropy is None:
            target_entropy = -act_dim
        self.target_entropy = target_entropy
        if alpha is None:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.automatic_entropy_tuning = True
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor([alpha], device=self.device)
            self.automatic_entropy_tuning = False

        self.total_it = 0

    # ------------------------------------------------------------------ action
    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.actor(
                obs, deterministic=deterministic, with_logprob=False
            )
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------ update
    def update(self, replay_buffer, batch_size=256):
        """Perform G gradient steps (G=1 by default). Returns logs dict."""
        logs = {}
        for _ in range(self.G):
            batch = replay_buffer.sample_batch(batch_size)
            obs = torch.as_tensor(batch["obs"], device=self.device)
            act = torch.as_tensor(batch["act"], device=self.device)
            rew = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device)
            obs2 = torch.as_tensor(batch["obs2"], device=self.device)
            done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

            # ---- Critic update ----
            with torch.no_grad():
                next_action, next_log_pi, _ = self.actor(
                    obs2, deterministic=False, with_logprob=True
                )
                # REDQ target: min over random subset of M target critics.
                target_q = self.critic_target.target_min(
                    obs2, next_action, self.M, rng=self.rng
                )
                target_q = rew + (1.0 - done) * self.gamma * (
                    target_q - self.alpha_exp() * next_log_pi
                )

            current_qs = self.critic(obs, act)  # (N, batch)
            critic_loss = 0.5 * sum(
                F.mse_loss(q, target_q) for q in current_qs
            ) / current_qs.shape[0]

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---- Actor update ----
            for p in self.critic.parameters():
                p.requires_grad = False
            pi_action, log_pi, _ = self.actor(
                obs, deterministic=False, with_logprob=True
            )
            # Actor uses mean Q over all N critics.
            q_pi = self.critic.mean_q(obs, pi_action)
            actor_loss = (self.alpha_exp() * log_pi - q_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for p in self.critic.parameters():
                p.requires_grad = True

            # ---- Entropy coefficient ----
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

            # ---- Soft target update ----
            with torch.no_grad():
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

            self.total_it += 1

        logs = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.alpha.detach()),
        }
        if self.automatic_entropy_tuning:
            logs["alpha_loss"] = float(alpha_loss.detach())
        return logs

    def alpha_exp(self):
        # self.alpha is already alpha = exp(log_alpha) when autotuning.
        return self.alpha

    # ------------------------------------------------------------------ save / load
    def save(self, path, step=None):
        """Save actor/critics + optimizer state for resumable training."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "step": step,
        }
        if self.automatic_entropy_tuning:
            payload["log_alpha"] = self.log_alpha.detach().clone()
            payload["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        else:
            payload["alpha"] = self.alpha.detach().clone()
        torch.save(payload, path)

    def load(self, path, device="cpu"):
        """Load a checkpoint and return it (callers can read ``step``)."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        # Optimizer state is optional for backwards compatibility with the
        # Phase 3/4 checkpoints saved before resumable training was added.
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.automatic_entropy_tuning:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.alpha = self.log_alpha.exp()
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        else:
            self.alpha = checkpoint["alpha"].to(device)
        self.total_it = checkpoint.get("total_it", 0)
        return checkpoint

    def uncertainty(self, obs, action):
        """Ensemble std of Q(s,a) across critics -> epistemic uncertainty U_t."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qs = self.critic.all_q(obs, act)  # (N, 1)
        return float(qs.std(unbiased=False).cpu().numpy()), float(qs.mean().cpu().numpy())
