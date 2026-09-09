"""Paper-aligned recurrent shared actor and centralized team critic."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from uav_swarm_control.models._distributions import diagonal_normal, squashed_log_probability
from uav_swarm_control.models.neighbor_encoder import (
    NeighborEncoderSpec,
    masked_mean_neighbor_features,
)


@dataclass(frozen=True, slots=True)
class RecurrentState:
    """LSTM hidden and cell tensors with layout ``[layers, batch, hidden]``."""

    hidden: Tensor
    cell: Tensor

    def detached(self) -> "RecurrentState":
        """Break the autograd history between rollout or TBPTT segments."""
        return RecurrentState(self.hidden.detach(), self.cell.detach())


@dataclass(frozen=True, slots=True)
class RecurrentPolicyOutput:
    """Actor quantities and the recurrent state after the last sequence item."""

    actions: Tensor
    latent_actions: Tensor
    log_probabilities: Tensor
    entropies: Tensor
    state: RecurrentState


def _orthogonal_initialize(module: nn.Module, *, output: nn.Linear) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.orthogonal_(child.weight, gain=2**0.5)
            nn.init.zeros_(child.bias)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    for child in module.modules():
        if isinstance(child, nn.LSTM):
            for name, parameter in child.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.zeros_(parameter)


class PaperRecurrentActorCritic(nn.Module):
    """FC-LSTM-FC actor/critic with shared actor weights and separate memories."""

    model_family = "paper-recurrent-mappo"

    def __init__(
        self,
        local_observation_size: int,
        centralized_state_size: int,
        *,
        action_size: int = 4,
        feature_size: int = 256,
        recurrent_hidden_size: int = 256,
        recurrent_layers: int = 1,
        initial_log_standard_deviation: float = -0.5,
        neighbor_encoder: NeighborEncoderSpec | None = None,
    ) -> None:
        super().__init__()
        dimensions = (
            local_observation_size,
            centralized_state_size,
            action_size,
            feature_size,
            recurrent_hidden_size,
            recurrent_layers,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("all recurrent model dimensions must be positive.")
        self.local_observation_size = local_observation_size
        self.centralized_state_size = centralized_state_size
        self.action_size = action_size
        self.feature_size = feature_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_layers = recurrent_layers
        self.initial_log_standard_deviation = initial_log_standard_deviation
        self.neighbor_encoder = neighbor_encoder

        actor_input_size = local_observation_size
        if neighbor_encoder is not None:
            if local_observation_size < neighbor_encoder.structured_width:
                raise ValueError("local observation is too narrow for the neighbor encoder.")
            self.neighbor_embedding = nn.Sequential(
                nn.Linear(neighbor_encoder.neighbor_features, neighbor_encoder.hidden_sizes[0]),
                nn.ReLU(),
                *[
                    layer
                    for left, right in zip(
                        neighbor_encoder.hidden_sizes,
                        neighbor_encoder.hidden_sizes[1:],
                        strict=False,
                    )
                    for layer in (nn.Linear(left, right), nn.ReLU())
                ],
                nn.Linear(neighbor_encoder.hidden_sizes[-1], neighbor_encoder.embedding_size),
            )
            actor_input_size = (
                neighbor_encoder.ego_features
                + neighbor_encoder.embedding_size
                + local_observation_size
                - neighbor_encoder.structured_width
            )
        else:
            self.neighbor_embedding = None

        self.actor_input = nn.Linear(actor_input_size, feature_size)
        self.actor_input_norm = nn.LayerNorm(feature_size)
        self.actor_lstm = nn.LSTM(feature_size, recurrent_hidden_size, recurrent_layers)
        self.actor_output_norm = nn.LayerNorm(recurrent_hidden_size)
        self.actor_output = nn.Linear(recurrent_hidden_size, action_size)
        self.log_standard_deviation = nn.Parameter(
            torch.full((action_size,), initial_log_standard_deviation)
        )

        self.critic_input = nn.Linear(centralized_state_size, feature_size)
        self.critic_input_norm = nn.LayerNorm(feature_size)
        self.critic_lstm = nn.LSTM(feature_size, recurrent_hidden_size, recurrent_layers)
        self.critic_output_norm = nn.LayerNorm(recurrent_hidden_size)
        self.critic_output = nn.Linear(recurrent_hidden_size, 1)

        _orthogonal_initialize(self, output=self.actor_output)
        nn.init.orthogonal_(self.critic_output.weight, gain=1.0)
        nn.init.zeros_(self.critic_output.bias)

    def initial_actor_state(self, batch_size: int, *, device: torch.device) -> RecurrentState:
        """Create independent zero memory for every flattened environment/UAV row."""
        return self._initial_state(batch_size, device=device)

    def initial_critic_state(self, batch_size: int, *, device: torch.device) -> RecurrentState:
        """Create one zero critic memory per environment."""
        return self._initial_state(batch_size, device=device)

    def _initial_state(self, batch_size: int, *, device: torch.device) -> RecurrentState:
        if batch_size < 1:
            raise ValueError("recurrent-state batch size must be positive.")
        values = torch.zeros(
            (self.recurrent_layers, batch_size, self.recurrent_hidden_size),
            dtype=torch.float32,
            device=device,
        )
        return RecurrentState(values, values.clone())

    def _validate_state(self, state: RecurrentState, batch_size: int) -> None:
        expected = (self.recurrent_layers, batch_size, self.recurrent_hidden_size)
        if state.hidden.shape != expected or state.cell.shape != expected:
            raise ValueError(f"recurrent state must contain hidden/cell tensors shaped {expected}.")

    @staticmethod
    def _apply_reset(state: RecurrentState, keep_mask: Tensor) -> RecurrentState:
        if keep_mask.ndim != 1:
            raise ValueError("recurrent keep mask must have shape [batch].")
        multiplier = keep_mask.to(dtype=state.hidden.dtype).view(1, -1, 1)
        return RecurrentState(state.hidden * multiplier, state.cell * multiplier)

    def _actor_features(self, observations: Tensor) -> Tensor:
        if observations.ndim != 3 or observations.shape[-1] != self.local_observation_size:
            raise ValueError("actor observations must have shape [time, batch, local_features].")
        inputs = observations
        if self.neighbor_encoder is not None:
            assert self.neighbor_embedding is not None
            inputs = masked_mean_neighbor_features(
                observations,
                self.neighbor_encoder,
                self.neighbor_embedding,
            )
        return self.actor_input_norm(torch.relu(self.actor_input(inputs)))

    def actor_means(
        self,
        observations: Tensor,
        state: RecurrentState,
        keep_masks: Tensor,
    ) -> tuple[Tensor, RecurrentState]:
        """Unroll the actor chronologically, resetting memory before marked items."""
        features = self._actor_features(observations)
        time_steps, batch_size, _ = features.shape
        if keep_masks.shape != (time_steps, batch_size):
            raise ValueError("actor keep masks must have shape [time, batch].")
        self._validate_state(state, batch_size)
        outputs: list[Tensor] = []
        current = state
        for index in range(time_steps):
            current = self._apply_reset(current, keep_masks[index])
            values, raw_state = self.actor_lstm(
                features[index : index + 1],
                (current.hidden, current.cell),
            )
            current = RecurrentState(*raw_state)
            outputs.append(self.actor_output(self.actor_output_norm(values)))
        return torch.cat(outputs, dim=0), current

    def sample_actions(
        self,
        observations: Tensor,
        state: RecurrentState,
        keep_masks: Tensor,
    ) -> RecurrentPolicyOutput:
        """Sample squashed Gaussian actions for a chronological sequence."""
        means, next_state = self.actor_means(observations, state, keep_masks)
        distribution = diagonal_normal(means, self.log_standard_deviation)
        latent = distribution.rsample()
        return RecurrentPolicyOutput(
            torch.tanh(latent),
            latent,
            squashed_log_probability(distribution, latent),
            distribution.entropy().sum(dim=-1),
            next_state,
        )

    def evaluate_actions(
        self,
        observations: Tensor,
        latent_actions: Tensor,
        state: RecurrentState,
        keep_masks: Tensor,
    ) -> RecurrentPolicyOutput:
        """Re-evaluate stored pre-tanh actions along their original sequence."""
        if latent_actions.shape != (*observations.shape[:-1], self.action_size):
            raise ValueError("latent actions must match observation sequence axes.")
        means, next_state = self.actor_means(observations, state, keep_masks)
        distribution = diagonal_normal(means, self.log_standard_deviation)
        return RecurrentPolicyOutput(
            torch.tanh(latent_actions),
            latent_actions,
            squashed_log_probability(distribution, latent_actions),
            distribution.entropy().sum(dim=-1),
            next_state,
        )

    def deterministic_actions(
        self,
        observations: Tensor,
        state: RecurrentState,
        keep_masks: Tensor,
    ) -> tuple[Tensor, RecurrentState]:
        """Return squashed means without sampling."""
        means, next_state = self.actor_means(observations, state, keep_masks)
        return torch.tanh(means), next_state

    def values(
        self,
        centralized_states: Tensor,
        state: RecurrentState,
        keep_masks: Tensor,
    ) -> tuple[Tensor, RecurrentState]:
        """Predict one cooperative team value per environment and time step."""
        if (
            centralized_states.ndim != 3
            or centralized_states.shape[-1] != self.centralized_state_size
        ):
            raise ValueError("critic states must have shape [time, batch, state_features].")
        time_steps, batch_size, _ = centralized_states.shape
        if keep_masks.shape != (time_steps, batch_size):
            raise ValueError("critic keep masks must have shape [time, batch].")
        self._validate_state(state, batch_size)
        features = self.critic_input_norm(torch.relu(self.critic_input(centralized_states)))
        outputs: list[Tensor] = []
        current = state
        for index in range(time_steps):
            current = self._apply_reset(current, keep_masks[index])
            values, raw_state = self.critic_lstm(
                features[index : index + 1],
                (current.hidden, current.cell),
            )
            current = RecurrentState(*raw_state)
            outputs.append(self.critic_output(self.critic_output_norm(values)).squeeze(-1))
        return torch.cat(outputs, dim=0), current
