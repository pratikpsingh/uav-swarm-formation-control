"""Compact actor and recurrent-state tests."""

import pytest
import torch

from uav_swarm_control.deployment.models import (
    CompactFeedForwardActor,
    CompactGRUActor,
    CompactLSTMActor,
)
from uav_swarm_control.models import NeighborEncoderSpec


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
