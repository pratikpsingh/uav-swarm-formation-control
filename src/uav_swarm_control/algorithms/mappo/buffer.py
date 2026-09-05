"""Shape-explicit rollout storage for multi-agent PPO."""

from dataclasses import dataclass

import torch
from torch import Tensor

from uav_swarm_control.algorithms.ppo.advantages import generalized_advantage_estimate


@dataclass(frozen=True, slots=True)
class FlatMAPPOBatch:
    """Agent samples flattened only after time/environment/agent validation."""

    local_observations: Tensor
    centralized_states: Tensor
    agent_indices: Tensor
    latent_actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor

    @property
    def size(self) -> int:
        """Number of agent-transition samples."""
        return self.local_observations.shape[0]


@dataclass(frozen=True, slots=True)
class MAPPOBatch:
    """One rollout preserving `[T, E, N, ...]` semantics."""

    local_observations: Tensor
    centralized_states: Tensor
    latent_actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor

    def flatten(self) -> FlatMAPPOBatch:
        """Create aligned agent samples for shuffled PPO minibatches."""
        time_steps, environments, agents, local_size = self.local_observations.shape
        state_size = self.centralized_states.shape[-1]
        action_size = self.latent_actions.shape[-1]
        expanded_states = self.centralized_states.unsqueeze(2).expand(
            time_steps,
            environments,
            agents,
            state_size,
        )
        agent_indices = (
            torch.arange(agents, device=self.local_observations.device)
            .view(1, 1, agents)
            .expand(time_steps, environments, agents)
        )
        return FlatMAPPOBatch(
            local_observations=self.local_observations.reshape(-1, local_size),
            centralized_states=expanded_states.reshape(-1, state_size),
            agent_indices=agent_indices.reshape(-1),
            latent_actions=self.latent_actions.reshape(-1, action_size),
            old_log_probabilities=self.old_log_probabilities.reshape(-1),
            old_values=self.old_values.reshape(-1),
            returns=self.returns.reshape(-1),
            advantages=self.advantages.reshape(-1),
        )


class MAPPORolloutBuffer:
    """Preallocated `[time, environment, agent]` on-policy storage."""

    def __init__(
        self,
        *,
        time_steps: int,
        num_environments: int,
        num_agents: int,
        local_observation_size: int,
        centralized_state_size: int,
        action_size: int,
        device: torch.device,
    ) -> None:
        dimensions = (
            time_steps,
            num_environments,
            num_agents,
            local_observation_size,
            centralized_state_size,
            action_size,
        )
        if any(dimension < 1 for dimension in dimensions):
            raise ValueError("all MAPPO rollout dimensions must be positive.")
        self.time_steps = time_steps
        self.num_environments = num_environments
        self.num_agents = num_agents
        self.local_observations = torch.empty(
            (time_steps, num_environments, num_agents, local_observation_size),
            device=device,
        )
        self.centralized_states = torch.empty(
            (time_steps, num_environments, centralized_state_size),
            device=device,
        )
        self.latent_actions = torch.empty(
            (time_steps, num_environments, num_agents, action_size),
            device=device,
        )
        agent_shape = (time_steps, num_environments, num_agents)
        self.rewards = torch.empty(agent_shape, device=device)
        self.log_probabilities = torch.empty(agent_shape, device=device)
        self.values = torch.empty(agent_shape, device=device)
        self.next_values = torch.empty(agent_shape, device=device)
        self.terminated = torch.empty(agent_shape, dtype=torch.bool, device=device)
        self.truncated = torch.empty(agent_shape, dtype=torch.bool, device=device)
        self._size = 0

    @property
    def full(self) -> bool:
        return self._size == self.time_steps

    def add(
        self,
        *,
        local_observations: Tensor,
        centralized_states: Tensor,
        latent_actions: Tensor,
        rewards: Tensor,
        log_probabilities: Tensor,
        values: Tensor,
        next_values: Tensor,
        terminated: Tensor,
        truncated: Tensor,
    ) -> None:
        """Append one complete environment-batch transition."""
        if self.full:
            raise RuntimeError("MAPPO rollout buffer is full.")
        supplied = (
            local_observations,
            centralized_states,
            latent_actions,
            rewards,
            log_probabilities,
            values,
            next_values,
            terminated,
            truncated,
        )
        destinations = (
            self.local_observations,
            self.centralized_states,
            self.latent_actions,
            self.rewards,
            self.log_probabilities,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
        )
        index = self._size
        for source, destination in zip(supplied, destinations, strict=True):
            if source.shape != destination.shape[1:]:
                raise ValueError(
                    f"rollout item has shape {source.shape}; expected {destination.shape[1:]}."
                )
            destination[index].copy_(source.detach())
        self._size += 1

    def as_batch(self, *, gamma: float, gae_lambda: float) -> MAPPOBatch:
        """Compute normalized per-agent advantages for a complete rollout."""
        if not self.full:
            raise RuntimeError("MAPPO rollout must be full before creating a batch.")
        advantages, returns = generalized_advantage_estimate(
            self.rewards,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return MAPPOBatch(
            local_observations=self.local_observations,
            centralized_states=self.centralized_states,
            latent_actions=self.latent_actions,
            old_log_probabilities=self.log_probabilities,
            old_values=self.values,
            returns=returns,
            advantages=advantages,
        )
