"""Tests for full MAPPO model checkpoint persistence."""

from pathlib import Path

import torch

from uav_swarm_control.algorithms.mappo import (
    load_mappo_checkpoint,
    save_mappo_checkpoint,
)
from uav_swarm_control.models import NeighborEncoderSpec, SharedActorCentralCritic


def test_mappo_checkpoint_reloads_actor_and_critic_identically(tmp_path: Path) -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    model = SharedActorCentralCritic(5, 9, 3, 3, (8, 8), (16, 16), -0.5)
    local = torch.randn(3, 5)
    state = torch.randn(2, 9)
    expected_actions = model.deterministic_actions(local)
    expected_values = model.values(state)
    path = tmp_path / "mappo.pt"

    save_mappo_checkpoint(path, model, metadata={"environment_steps": 128})
    loaded, metadata = load_mappo_checkpoint(path)

    torch.testing.assert_close(loaded.deterministic_actions(local), expected_actions)
    torch.testing.assert_close(loaded.values(state), expected_values)
    assert metadata == {"environment_steps": 128}


def test_checkpoint_preserves_neighbor_encoder_architecture(tmp_path: Path) -> None:
    model = SharedActorCentralCritic(
        35,
        12,
        3,
        4,
        (8, 8),
        (16, 16),
        -0.5,
        neighbor_encoder=NeighborEncoderSpec(6, 3, 6, 8, (12,)),
    )
    local = torch.randn(4, 35)
    expected = model.deterministic_actions(local)
    path = tmp_path / "neighbor-mappo.pt"

    save_mappo_checkpoint(path, model)
    loaded, _ = load_mappo_checkpoint(path)

    assert loaded.neighbor_encoder == model.neighbor_encoder
    torch.testing.assert_close(loaded.deterministic_actions(local), expected)
