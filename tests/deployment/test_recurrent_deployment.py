"""Actor-only extraction and export tests for recurrent deployment."""

from pathlib import Path

import torch

from uav_swarm_control.deployment.artifacts import count_parameters, export_actor
from uav_swarm_control.deployment.contracts import (
    DeploymentActionProfile,
    DeploymentArchitecture,
)
from uav_swarm_control.evaluation.recurrent_deployment import RecurrentTeacherActor
from uav_swarm_control.models import NeighborEncoderSpec, PaperRecurrentActorCritic


def test_recurrent_teacher_wrapper_excludes_critic_and_matches_actor(tmp_path: Path) -> None:
    specification = NeighborEncoderSpec(
        ego_features=6,
        max_neighbors=2,
        neighbor_features=6,
        embedding_size=4,
        hidden_sizes=(5,),
    )
    model = PaperRecurrentActorCritic(
        20,
        36,
        feature_size=8,
        recurrent_hidden_size=8,
        neighbor_encoder=specification,
    ).eval()
    actor = RecurrentTeacherActor(model).eval()
    observations = torch.randn(3, 4, 20)
    state = model.initial_actor_state(3, device=torch.device("cpu"))
    keep = torch.ones(4, 3, dtype=torch.bool)

    with torch.no_grad():
        expected, expected_state = model.deterministic_actions(
            observations.permute(1, 0, 2),
            state,
            keep,
        )
        actual, hidden, cell = actor(observations, state.hidden, state.cell)

    torch.testing.assert_close(actual, expected.permute(1, 0, 2))
    torch.testing.assert_close(hidden, expected_state.hidden)
    torch.testing.assert_close(cell, expected_state.cell)
    assert count_parameters(actor) < count_parameters(model)
    assert all("critic" not in name for name in actor.state_dict())

    artifact, metadata = export_actor(
        actor,
        tmp_path / "teacher",
        architecture=DeploymentArchitecture.LSTM,
        observation_size=20,
        action_size=4,
        recurrent_width=8,
        candidate="teacher",
        quantization="none",
        structured_zero_channel_fraction=0.0,
        action_profile=DeploymentActionProfile.DIRECTION_SPEED,
    )
    assert artifact.is_file()
    assert metadata["contains_centralized_critic"] is False
    assert metadata["action_profile"] == "direction-speed"
    assert metadata["logical_recurrent_state_bytes_per_agent"] == 64
