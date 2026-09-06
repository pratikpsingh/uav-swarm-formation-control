"""Episode-level distillation and structured-pruning tests."""

import torch

from uav_swarm_control.configuration.deployment import (
    CompressionCandidateConfig,
    DistillationConfig,
)
from uav_swarm_control.deployment.contracts import DeploymentArchitecture
from uav_swarm_control.deployment.distillation import TeacherTrajectory, distill_student
from uav_swarm_control.models import NeighborEncoderSpec


def _trajectories() -> tuple[TeacherTrajectory, ...]:
    result: list[TeacherTrajectory] = []
    for episode, length in enumerate((3, 4, 5, 6)):
        observations = torch.linspace(-1.0, 1.0, length * 2 * 10).reshape(length, 2, 10)
        actions = torch.tanh(observations[..., :3])
        result.append(TeacherTrajectory(observations, actions, "condition", episode))
    return tuple(result)


def test_distillation_splits_complete_episodes_and_retains_structured_zeros() -> None:
    candidate = CompressionCandidateConfig(
        "pruned", DeploymentArchitecture.FEED_FORWARD, 8, 2, 0.5, "none"
    )
    result = distill_student(
        candidate,
        _trajectories(),
        DistillationConfig(2, 0.25, 2, 1, 4, 0.01),
        seed=7,
        local_observation_size=10,
        action_size=3,
        teacher_neighbor_spec=NeighborEncoderSpec(2, 2, 2, 4, (4,)),
    )
    assert result.training_trajectories == 3
    assert result.validation_trajectories == 1
    assert result.structured_zero_channel_fraction > 0.0
    assert result.training_mse >= 0.0
    assert result.validation_mse >= 0.0
    hidden_linears = [
        module
        for module in result.model.modules()
        if isinstance(module, torch.nn.Linear) and module.out_features != 3
    ]
    assert any(bool((module.weight == 0.0).all(dim=1).any()) for module in hidden_linears)
