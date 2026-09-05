"""Parameter-shared decentralized actor with a centralized training critic."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from uav_swarm_control.models._distributions import (
    diagonal_normal,
    squashed_log_probability,
)
from uav_swarm_control.models._networks import build_mlp


@dataclass(frozen=True, slots=True)
class SharedPolicyOutput:
    """Per-agent quantities produced by the one shared actor."""

    actions: Tensor
    latent_actions: Tensor
    log_probabilities: Tensor
    entropies: Tensor


class SharedActorCentralCritic(nn.Module):
    """Keep decentralized actor inputs structurally separate from privileged critic state."""

    def __init__(
        self,
        local_observation_size: int,
        centralized_state_size: int,
        action_size: int,
        num_agents: int,
        actor_hidden_sizes: tuple[int, ...],
        critic_hidden_sizes: tuple[int, ...],
        initial_log_standard_deviation: float,
    ) -> None:
        super().__init__()
        if min(local_observation_size, centralized_state_size, action_size, num_agents) < 1:
            raise ValueError("model dimensions must be positive.")
        self.local_observation_size = local_observation_size
        self.centralized_state_size = centralized_state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.actor_hidden_sizes = actor_hidden_sizes
        self.critic_hidden_sizes = critic_hidden_sizes
        self.initial_log_standard_deviation = initial_log_standard_deviation
        self.actor = build_mlp(
            local_observation_size,
            action_size,
            actor_hidden_sizes,
            output_gain=0.01,
        )
        self.critic = build_mlp(
            centralized_state_size,
            num_agents,
            critic_hidden_sizes,
            output_gain=1.0,
        )
        self.log_standard_deviation = nn.Parameter(
            torch.full((action_size,), initial_log_standard_deviation)
        )

    def _validate_local(self, local_observations: Tensor) -> None:
        if (
            local_observations.ndim < 2
            or local_observations.shape[-1] != self.local_observation_size
        ):
            raise ValueError(
                "local observations must have at least two dimensions and the configured width."
            )

    def sample_actions(self, local_observations: Tensor) -> SharedPolicyOutput:
        """Sample one bounded action per local row using shared actor parameters."""
        self._validate_local(local_observations)
        distribution = diagonal_normal(
            self.actor(local_observations),
            self.log_standard_deviation,
        )
        latent_actions = distribution.rsample()
        return SharedPolicyOutput(
            actions=torch.tanh(latent_actions),
            latent_actions=latent_actions,
            log_probabilities=squashed_log_probability(distribution, latent_actions),
            entropies=distribution.entropy().sum(dim=-1),
        )

    def evaluate_latent_actions(
        self,
        local_observations: Tensor,
        latent_actions: Tensor,
    ) -> SharedPolicyOutput:
        """Recompute densities for losslessly stored pre-tanh actions."""
        self._validate_local(local_observations)
        if latent_actions.shape != (*local_observations.shape[:-1], self.action_size):
            raise ValueError("latent action leading dimensions must match local observations.")
        distribution = diagonal_normal(
            self.actor(local_observations),
            self.log_standard_deviation,
        )
        return SharedPolicyOutput(
            actions=torch.tanh(latent_actions),
            latent_actions=latent_actions,
            log_probabilities=squashed_log_probability(distribution, latent_actions),
            entropies=distribution.entropy().sum(dim=-1),
        )

    def deterministic_actions(self, local_observations: Tensor) -> Tensor:
        """Return bounded actor means without access to centralized state."""
        self._validate_local(local_observations)
        return torch.tanh(self.actor(local_observations))

    def values(self, centralized_states: Tensor) -> Tensor:
        """Predict one value per agent from training-only centralized state."""
        if (
            centralized_states.ndim < 2
            or centralized_states.shape[-1] != self.centralized_state_size
        ):
            raise ValueError(
                "centralized states must have at least two dimensions and the configured width."
            )
        return self.critic(centralized_states)

    def values_for_agents(self, centralized_states: Tensor, agent_indices: Tensor) -> Tensor:
        """Select the correct per-agent critic output for flattened samples."""
        values = self.values(centralized_states)
        if values.ndim != 2 or agent_indices.shape != values.shape[:1]:
            raise ValueError("flattened agent indices must contain one entry per state row.")
        if agent_indices.dtype != torch.int64:
            raise TypeError("agent indices must use torch.int64.")
        return values.gather(1, agent_indices.unsqueeze(1)).squeeze(1)
