from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class TrajectoryBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    action_mask: Optional[torch.Tensor] = None


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=rewards.dtype)
    next_value = torch.tensor(0.0, dtype=rewards.dtype)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1.0 - dones[step]) * gae
        advantages[step] = gae
        next_value = values[step]
    returns = advantages + values
    return returns, advantages


class GaussianActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.6))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        mean = torch.tanh(self.mean_head(hidden))
        std = self.log_std.exp().expand_as(mean)
        value = self.value_head(hidden).squeeze(-1)
        return mean, std, value

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = torch.clamp(dist.rsample(), -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, values = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values


class CategoricalActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        logits = self.policy_head(hidden)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def sample(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state, action_mask=action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(states, action_mask=action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrajectoryBatch,
    clip_ratio: float = 0.2,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    epochs: int = 4,
) -> dict:
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    last_loss = {}
    for _ in range(epochs):
        if isinstance(model, GaussianActorCritic):
            log_probs, entropy, values = model.evaluate_actions(batch.states, batch.actions)
        else:
            log_probs, entropy, values = model.evaluate_actions(
                batch.states,
                batch.actions.long(),
                action_mask=batch.action_mask,
            )
        ratio = torch.exp(log_probs - batch.old_log_probs)
        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
        value_loss = (batch.returns - values).pow(2).mean()
        entropy_bonus = entropy.mean()
        loss = policy_loss + value_weight * value_loss - entropy_weight * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }
    return last_loss
