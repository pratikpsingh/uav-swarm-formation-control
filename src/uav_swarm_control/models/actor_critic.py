"""Continuous actor and critic used by the PPO reference implementation."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as functional


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Quantities recorded for a sampled on-policy transition."""

    action: Tensor
    latent_action: Tensor
    log_probability: Tensor
    entropy: Tensor
    value: Tensor


def _mlp(input_size: int, output_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layer = nn.Linear(previous, size)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        previous = size
    output = nn.Linear(previous, output_size)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


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
        self.actor = _mlp(observation_size, action_size, hidden_sizes)
        self.critic = _mlp(observation_size, 1, hidden_sizes)
        critic_output = self.critic[-1]
        if isinstance(critic_output, nn.Linear):
            nn.init.orthogonal_(critic_output.weight, gain=1.0)
        self.log_standard_deviation = nn.Parameter(
            torch.full((action_size,), initial_log_standard_deviation)
        )

    def _distribution(self, observations: Tensor) -> Normal:
        means = self.actor(observations)
        standard_deviations = self.log_standard_deviation.exp().expand_as(means)
        return Normal(means, standard_deviations)

    @staticmethod
    def _squashed_log_probability(
        distribution: Normal,
        latent: Tensor,
    ) -> Tensor:
        log_jacobian = 2.0 * (math.log(2.0) - latent - functional.softplus(-2.0 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(dim=-1)

    def sample(self, observations: Tensor) -> PolicyOutput:
        """Sample tanh-squashed actions and return corrected log probabilities."""
        distribution = self._distribution(observations)
        latent = distribution.rsample()
        action = torch.tanh(latent)
        return PolicyOutput(
            action=action,
            latent_action=latent,
            log_probability=self._squashed_log_probability(distribution, latent),
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
            log_probability=self._squashed_log_probability(distribution, latent_actions),
            entropy=distribution.entropy().sum(dim=-1),
            value=self.value(observations),
        )

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """Use the bounded distribution mean for evaluation or deployment."""
        return torch.tanh(self.actor(observations))

    def value(self, observations: Tensor) -> Tensor:
        """Estimate expected return for each observation."""
        return self.critic(observations).squeeze(-1)
