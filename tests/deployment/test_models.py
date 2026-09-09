"""Compact actor and recurrent-state tests."""

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from uav_swarm_control.agents import sequential_agent_ids
from uav_swarm_control.deployment.contracts import (
    DeploymentActionProfile,
    DeploymentArchitecture,
)
from uav_swarm_control.deployment.models import (
    CompactFeedForwardActor,
    CompactGRUActor,
    CompactLSTMActor,
    DeploymentController,
)
from uav_swarm_control.models import NeighborEncoderSpec
from uav_swarm_control.observations import LocalObservations


def _spec() -> NeighborEncoderSpec:
    return NeighborEncoderSpec(
        ego_features=2, max_neighbors=2, neighbor_features=2, embedding_size=4, hidden_sizes=(4,)
    )


def test_compact_feed_forward_retains_neighbor_permutation_invariance() -> None:
    model = CompactFeedForwardActor(10, 3, 8, 2, _spec()).eval()
    observation = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 7.0, 8.0]])
    permuted = observation[:, [0, 1, 4, 5, 2, 3, 7, 6, 8, 9]]
    with torch.no_grad():
        assert torch.allclose(model(observation), model(permuted), atol=1e-6)


@pytest.mark.parametrize("kind", ["gru", "lstm"])
def test_recurrent_actors_make_state_explicit(kind: str) -> None:
    observations = torch.zeros(3, 5, 10)
    hidden = torch.zeros(1, 3, 6)
    if kind == "gru":
        actions, next_hidden = CompactGRUActor(10, 3, 6, _spec())(observations, hidden)
        assert actions.shape == (3, 5, 3)
        assert next_hidden.shape == hidden.shape
    else:
        actions, next_hidden, next_cell = CompactLSTMActor(10, 3, 6, _spec())(
            observations, hidden, torch.zeros_like(hidden)
        )
        assert actions.shape == (3, 5, 3)
        assert next_hidden.shape == next_cell.shape == hidden.shape
    assert bool(torch.all(actions.abs() <= 1.0))


class ConstantDirectionActor(nn.Module):
    def forward(self, observations: Tensor) -> Tensor:
        action = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=observations.dtype)
        return action.expand(observations.shape[0], -1)


def test_deployment_controller_converts_direction_and_speed_to_velocity() -> None:
    observations = LocalObservations(
        sequential_agent_ids(2),
        np.zeros((2, 6), dtype=np.float32),
        np.zeros((2, 0, 6), dtype=np.float32),
        np.zeros((2, 0), dtype=np.bool_),
    )
    controller = DeploymentController(
        ConstantDirectionActor(),
        DeploymentArchitecture.FEED_FORWARD,
        action_profile=DeploymentActionProfile.DIRECTION_SPEED,
    )

    actions = controller.act(observations)

    np.testing.assert_array_equal(
        actions.values,
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
