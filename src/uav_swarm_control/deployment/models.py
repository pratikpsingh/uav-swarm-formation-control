"""Small decentralized actors with explicit recurrent-state boundaries."""

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from uav_swarm_control.deployment.contracts import DeploymentArchitecture
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.models._networks import build_mlp
from uav_swarm_control.models.neighbor_encoder import (
    NeighborEncoderSpec,
    masked_mean_neighbor_features,
)
from uav_swarm_control.observations import LocalObservations, encode_local_observations


class CompactNeighborFeatures(nn.Module):
    """Smaller version of the teacher's permutation-invariant feature extractor."""

    def __init__(self, local_observation_size: int, width: int, spec: NeighborEncoderSpec) -> None:
        super().__init__()
        if local_observation_size < spec.structured_width:
            raise ValueError("local observation is too narrow for the neighbor layout.")
        self.spec = spec
        self.neighbor_encoder = build_mlp(
            spec.neighbor_features, spec.embedding_size, (width,), output_gain=1.0
        )
        self.output_size = (
            spec.ego_features + spec.embedding_size + local_observation_size - spec.structured_width
        )

    def forward(self, observations: Tensor) -> Tensor:
        return masked_mean_neighbor_features(observations, self.spec, self.neighbor_encoder)


class CompactFeedForwardActor(nn.Module):
    """Masked-mean actor whose capacity is controlled by one width and depth."""

    def __init__(
        self,
        local_observation_size: int,
        action_size: int,
        width: int,
        depth: int,
        neighbor_spec: NeighborEncoderSpec,
    ) -> None:
        super().__init__()
        if min(action_size, width, depth) < 1:
            raise ValueError("actor action size, width, and depth must be positive.")
        self.features = CompactNeighborFeatures(local_observation_size, width, neighbor_spec)
        self.policy = build_mlp(
            self.features.output_size, action_size, (width,) * depth, output_gain=0.01
        )

    def forward(self, observations: Tensor) -> Tensor:
        return torch.tanh(self.policy(self.features(observations)))


class CompactGRUActor(nn.Module):
    """Sequence actor with caller-owned GRU state for decentralized deployment."""

    def __init__(
        self,
        local_observation_size: int,
        action_size: int,
        width: int,
        neighbor_spec: NeighborEncoderSpec,
    ) -> None:
        super().__init__()
        if min(action_size, width) < 1:
            raise ValueError("actor action size and width must be positive.")
        self.width = width
        self.features = CompactNeighborFeatures(local_observation_size, width, neighbor_spec)
        self.recurrent = nn.GRU(self.features.output_size, width, batch_first=True)
        self.output = nn.Linear(width, action_size)

    def forward(self, observations: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.features(observations)
        recurrent, next_hidden = self.recurrent(encoded, hidden)
        return torch.tanh(self.output(recurrent)), next_hidden


class CompactLSTMActor(nn.Module):
    """Sequence actor with caller-owned LSTM hidden and cell state."""

    def __init__(
        self,
        local_observation_size: int,
        action_size: int,
        width: int,
        neighbor_spec: NeighborEncoderSpec,
    ) -> None:
        super().__init__()
        if min(action_size, width) < 1:
            raise ValueError("actor action size and width must be positive.")
        self.width = width
        self.features = CompactNeighborFeatures(local_observation_size, width, neighbor_spec)
        self.recurrent = nn.LSTM(self.features.output_size, width, batch_first=True)
        self.output = nn.Linear(width, action_size)

    def forward(
        self, observations: Tensor, hidden: Tensor, cell: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.features(observations)
        recurrent, (next_hidden, next_cell) = self.recurrent(encoded, (hidden, cell))
        return torch.tanh(self.output(recurrent)), next_hidden, next_cell


DeploymentActor = CompactFeedForwardActor | CompactGRUActor | CompactLSTMActor


class DeploymentController:
    """Adapt compact actors to the environment and reset state at episode boundaries."""

    def __init__(self, model: nn.Module, architecture: DeploymentArchitecture) -> None:
        self.model = model
        self.architecture = architecture
        self._hidden: Tensor | None = None
        self._cell: Tensor | None = None

    def reset(self) -> None:
        self._hidden = None
        self._cell = None

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        values = torch.as_tensor(
            np.array(encode_local_observations(observations), copy=True), dtype=torch.float32
        )
        with torch.no_grad():
            if self.architecture is DeploymentArchitecture.FEED_FORWARD:
                actions = cast(CompactFeedForwardActor, self.model)(values)
            else:
                sequence = values.unsqueeze(1)
                recurrent = cast(CompactGRUActor | CompactLSTMActor, self.model)
                if self._hidden is None:
                    self._hidden = torch.zeros(1, len(values), recurrent.width)
                if self.architecture is DeploymentArchitecture.GRU:
                    output, self._hidden = cast(CompactGRUActor, recurrent)(sequence, self._hidden)
                else:
                    if self._cell is None:
                        self._cell = torch.zeros_like(self._hidden)
                    output, self._hidden, self._cell = cast(CompactLSTMActor, recurrent)(
                        sequence, self._hidden, self._cell
                    )
                actions = output[:, 0]
        array = actions.detach().cpu().numpy().astype(np.float32)
        return NormalizedVelocityActions(observations.agent_ids, array)
