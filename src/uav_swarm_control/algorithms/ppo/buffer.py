"""Fixed-size on-policy rollout storage."""

from dataclasses import dataclass

import torch
from torch import Tensor

from uav_swarm_control.algorithms.ppo.advantages import generalized_advantage_estimate


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """A complete rollout with computed learning targets."""

    observations: Tensor
    latent_actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor


class RolloutBuffer:
    """Preallocated transition storage whose shape errors fail immediately."""

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        action_size: int,
        device: torch.device,
    ) -> None:
        if capacity < 1 or observation_size < 1 or action_size < 1:
            raise ValueError("buffer dimensions must be positive.")
        self.capacity = capacity
        self.observations = torch.empty((capacity, observation_size), device=device)
        self.latent_actions = torch.empty((capacity, action_size), device=device)
        self.rewards = torch.empty(capacity, device=device)
        self.log_probabilities = torch.empty(capacity, device=device)
        self.values = torch.empty(capacity, device=device)
        self.next_values = torch.empty(capacity, device=device)
        self.terminated = torch.empty(capacity, dtype=torch.bool, device=device)
        self.truncated = torch.empty(capacity, dtype=torch.bool, device=device)
        self._size = 0

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    def add(
        self,
        *,
        observation: Tensor,
        latent_action: Tensor,
        reward: float,
        log_probability: Tensor,
        value: Tensor,
        next_value: Tensor,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one transition without retaining an autograd graph."""
        if self.full:
            raise RuntimeError("rollout buffer is full.")
        if observation.shape != self.observations.shape[1:]:
            raise ValueError("observation shape does not match the rollout buffer.")
        if latent_action.shape != self.latent_actions.shape[1:]:
            raise ValueError("latent action shape does not match the rollout buffer.")
        index = self._size
        self.observations[index].copy_(observation.detach())
        self.latent_actions[index].copy_(latent_action.detach())
        self.rewards[index] = reward
        self.log_probabilities[index].copy_(log_probability.detach())
        self.values[index].copy_(value.detach())
        self.next_values[index].copy_(next_value.detach())
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self._size += 1

    def as_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        normalize_advantages: bool = True,
    ) -> RolloutBatch:
        """Compute learning targets after a complete rollout."""
        if not self.full:
            raise RuntimeError("rollout buffer must be full before creating a batch.")
        advantages, returns = generalized_advantage_estimate(
            self.rewards,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return RolloutBatch(
            observations=self.observations,
            latent_actions=self.latent_actions,
            old_log_probabilities=self.log_probabilities,
            old_values=self.values,
            returns=returns,
            advantages=advantages,
        )
