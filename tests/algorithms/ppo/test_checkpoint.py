"""Tests for versioned policy checkpoint persistence."""

from pathlib import Path

import torch

from uav_swarm_control.algorithms.ppo import (
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from uav_swarm_control.models import ActorCritic


def test_saved_policy_reloads_with_identical_outputs(tmp_path: Path) -> None:
    torch.manual_seed(9)  # pyright: ignore[reportUnknownMemberType]
    model = ActorCritic(2, 1, (8, 4), -0.5)
    observations = torch.tensor([[-0.3, 0.2], [0.4, 0.1]])
    expected = model.deterministic_action(observations)
    path = tmp_path / "policy.pt"

    save_policy_checkpoint(path, model, metadata={"seed": 9, "environment_steps": 128})
    loaded, metadata = load_policy_checkpoint(path)

    torch.testing.assert_close(loaded.deterministic_action(observations), expected)
    assert metadata == {"seed": 9, "environment_steps": 128}
    assert loaded.hidden_sizes == (8, 4)
