"""Continuous actor and critic used by the PPO reference implementation."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from uav_swarm_control.models._distributions import (
    diagonal_normal,
    squashed_log_probability,
)
from uav_swarm_control.models._networks import build_mlp


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Quantities recorded for a sampled on-policy transition."""

    action: Tensor
    latent_action: Tensor
    log_probability: Tensor
    entropy: Tensor
    value: Tensor


class ActorCritic(nn.Module):
    """Separate policy and value networks for bounded continuous actions."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...],
        initial_log_standard_deviation: float,
    ) -> None:
        super().__init__()
        if observation_size < 1 or action_size < 1 or not hidden_sizes:
            raise ValueError("network dimensions must be positive.")
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.initial_log_standard_deviation = initial_log_standard_deviation
        self.actor = build_mlp(observation_size, action_size, hidden_sizes, output_gain=0.01)
        self.critic = build_mlp(observation_size, 1, hidden_sizes, output_gain=1.0)
        self.log_standard_deviation = nn.Parameter(
            torch.full((action_size,), initial_log_standard_deviation)
        )

    def _distribution(self, observations: Tensor) -> Normal:
        means = self.actor(observations)
        return diagonal_normal(means, self.log_standard_deviation)

    def sample(self, observations: Tensor) -> PolicyOutput:
        """Sample tanh-squashed actions and return corrected log probabilities."""
        distribution = self._distribution(observations)
        latent = distribution.rsample()
        action = torch.tanh(latent)
        return PolicyOutput(
            action=action,
            latent_action=latent,
            log_probability=squashed_log_probability(distribution, latent),
            entropy=distribution.entropy().sum(dim=-1),
            value=self.value(observations),
        )

    def evaluate_latent_actions(
        self,
        observations: Tensor,
        latent_actions: Tensor,
    ) -> PolicyOutput:
        """Recompute policy quantities from losslessly stored latent actions."""
        distribution = self._distribution(observations)
        actions = torch.tanh(latent_actions)
        return PolicyOutput(
            action=actions,
            latent_action=latent_actions,
            log_probability=squashed_log_probability(distribution, latent_actions),
            entropy=distribution.entropy().sum(dim=-1),
            value=self.value(observations),
        )

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """Use the bounded distribution mean for evaluation or deployment."""
        return torch.tanh(self.actor(observations))

    def value(self, observations: Tensor) -> Tensor:
        """Estimate expected return for each observation."""
        return self.critic(observations).squeeze(-1)
