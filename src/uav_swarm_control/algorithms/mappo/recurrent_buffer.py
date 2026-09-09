"""Chronological rollout storage for recurrent cooperative MAPPO."""

from dataclasses import dataclass

import torch
from torch import Tensor

from uav_swarm_control.algorithms.ppo.advantages import generalized_advantage_estimate
from uav_swarm_control.models.recurrent_actor_critic import RecurrentState


@dataclass(frozen=True, slots=True)
class RecurrentSequenceBatch:
    """Contiguous TBPTT sequences with one actor memory per UAV."""

    local_observations: Tensor  # [B,L,N,F]
    centralized_states: Tensor  # [B,L,S]
    latent_actions: Tensor  # [B,L,N,A]
    old_log_probabilities: Tensor  # [B,L,N]
    returns: Tensor  # [B,L]
    advantages: Tensor  # [B,L]
    keep_masks: Tensor  # [B,L]
    actor_initial_state: RecurrentState  # [R,B,N,H]
    critic_initial_state: RecurrentState  # [R,B,H]

    @property
    def sequence_count(self) -> int:
        return self.local_observations.shape[0]

    def select(self, indices: Tensor) -> "RecurrentSequenceBatch":
        """Select complete sequences and flatten their independent UAV memories."""
        actor_hidden = self.actor_initial_state.hidden[:, indices]
        actor_cell = self.actor_initial_state.cell[:, indices]
        layers, sequences, agents, hidden = actor_hidden.shape
        return RecurrentSequenceBatch(
            self.local_observations[indices],
            self.centralized_states[indices],
            self.latent_actions[indices],
            self.old_log_probabilities[indices],
            self.returns[indices],
            self.advantages[indices],
            self.keep_masks[indices],
            RecurrentState(
                actor_hidden.reshape(layers, sequences * agents, hidden),
                actor_cell.reshape(layers, sequences * agents, hidden),
            ),
            RecurrentState(
                self.critic_initial_state.hidden[:, indices],
                self.critic_initial_state.cell[:, indices],
            ),
        )

    def actor_inputs(self) -> tuple[Tensor, Tensor]:
        """Create time-major actor input with sequence and UAV axes flattened."""
        sequences, time_steps, agents, features = self.local_observations.shape
        observations = self.local_observations.permute(1, 0, 2, 3).reshape(
            time_steps, sequences * agents, features
        )
        masks = (
            self.keep_masks.permute(1, 0)
            .unsqueeze(-1)
            .expand(time_steps, sequences, agents)
            .reshape(time_steps, sequences * agents)
        )
        return observations, masks

    def critic_inputs(self) -> tuple[Tensor, Tensor]:
        return self.centralized_states.permute(1, 0, 2), self.keep_masks.permute(1, 0)


class RecurrentMAPPORolloutBuffer:
    """Preallocated rollout that records pre-step LSTM state at every time."""

    def __init__(
        self,
        *,
        time_steps: int,
        num_environments: int,
        num_agents: int,
        local_observation_size: int,
        centralized_state_size: int,
        action_size: int,
        recurrent_layers: int,
        recurrent_hidden_size: int,
        device: torch.device,
    ) -> None:
        dimensions = (
            time_steps,
            num_environments,
            num_agents,
            local_observation_size,
            centralized_state_size,
            action_size,
            recurrent_layers,
            recurrent_hidden_size,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("all recurrent rollout dimensions must be positive.")
        self.time_steps = time_steps
        self.num_environments = num_environments
        self.num_agents = num_agents
        self.local_observations = torch.empty(
            (time_steps, num_environments, num_agents, local_observation_size), device=device
        )
        self.centralized_states = torch.empty(
            (time_steps, num_environments, centralized_state_size), device=device
        )
        self.latent_actions = torch.empty(
            (time_steps, num_environments, num_agents, action_size), device=device
        )
        self.log_probabilities = torch.empty(
            (time_steps, num_environments, num_agents), device=device
        )
        team = (time_steps, num_environments)
        self.rewards = torch.empty(team, device=device)
        self.values = torch.empty(team, device=device)
        self.next_values = torch.empty(team, device=device)
        self.terminated = torch.empty(team, dtype=torch.bool, device=device)
        self.truncated = torch.empty(team, dtype=torch.bool, device=device)
        self.keep_masks = torch.empty(team, dtype=torch.bool, device=device)
        self.actor_hidden = torch.empty(
            (
                time_steps,
                recurrent_layers,
                num_environments,
                num_agents,
                recurrent_hidden_size,
            ),
            device=device,
        )
        self.actor_cell = torch.empty_like(self.actor_hidden)
        self.critic_hidden = torch.empty(
            (time_steps, recurrent_layers, num_environments, recurrent_hidden_size),
            device=device,
        )
        self.critic_cell = torch.empty_like(self.critic_hidden)
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
        keep_masks: Tensor,
        actor_state: RecurrentState,
        critic_state: RecurrentState,
    ) -> None:
        """Append a full parallel transition and its pre-step memories."""
        if self.full:
            raise RuntimeError("recurrent rollout buffer is full.")
        index = self._size
        sources = (
            local_observations,
            centralized_states,
            latent_actions,
            rewards,
            log_probabilities,
            values,
            next_values,
            terminated,
            truncated,
            keep_masks,
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
            self.keep_masks,
        )
        for source, destination in zip(sources, destinations, strict=True):
            if source.shape != destination.shape[1:]:
                raise ValueError(
                    f"rollout item has shape {source.shape}; expected {destination.shape[1:]}."
                )
            destination[index].copy_(source.detach())
        layers = self.actor_hidden.shape[1]
        hidden = self.actor_hidden.shape[-1]
        actor_shape = (layers, self.num_environments, self.num_agents, hidden)
        actor_hidden = actor_state.hidden.reshape(actor_shape)
        actor_cell = actor_state.cell.reshape(actor_shape)
        critic_shape = (layers, self.num_environments, hidden)
        if critic_state.hidden.shape != critic_shape or critic_state.cell.shape != critic_shape:
            raise ValueError(f"critic state must have shape {critic_shape}.")
        self.actor_hidden[index].copy_(actor_hidden.detach())
        self.actor_cell[index].copy_(actor_cell.detach())
        self.critic_hidden[index].copy_(critic_state.hidden.detach())
        self.critic_cell[index].copy_(critic_state.cell.detach())
        self._size += 1

    def as_sequences(
        self, *, sequence_length: int, gamma: float, gae_lambda: float
    ) -> RecurrentSequenceBatch:
        """Calculate team GAE and split each environment into contiguous chunks."""
        if not self.full:
            raise RuntimeError("recurrent rollout must be full before batching.")
        if sequence_length < 1 or self.time_steps % sequence_length:
            raise ValueError("sequence_length must divide rollout time_steps.")
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
        chunks = [
            (environment, start)
            for environment in range(self.num_environments)
            for start in range(0, self.time_steps, sequence_length)
        ]

        def temporal(values: Tensor) -> Tensor:
            return torch.stack(
                [
                    values[start : start + sequence_length, environment]
                    for environment, start in chunks
                ]
            )

        return RecurrentSequenceBatch(
            temporal(self.local_observations),
            temporal(self.centralized_states),
            temporal(self.latent_actions),
            temporal(self.log_probabilities),
            temporal(returns),
            temporal(advantages),
            temporal(self.keep_masks),
            RecurrentState(
                torch.stack(
                    [self.actor_hidden[start, :, environment] for environment, start in chunks],
                    dim=1,
                ),
                torch.stack(
                    [self.actor_cell[start, :, environment] for environment, start in chunks],
                    dim=1,
                ),
            ),
            RecurrentState(
                torch.stack(
                    [self.critic_hidden[start, :, environment] for environment, start in chunks],
                    dim=1,
                ),
                torch.stack(
                    [self.critic_cell[start, :, environment] for environment, start in chunks],
                    dim=1,
                ),
            ),
        )


__all__ = ["RecurrentMAPPORolloutBuffer", "RecurrentSequenceBatch"]
