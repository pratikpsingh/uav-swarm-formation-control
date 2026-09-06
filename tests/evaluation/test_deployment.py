"""Teacher evidence gate tests for Stage 12."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from uav_swarm_control.algorithms.mappo import save_mappo_checkpoint
from uav_swarm_control.configuration.communication import load_communication_experiment_config
from uav_swarm_control.configuration.deployment import load_deployment_study_config
from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.deployment import validate_teacher
from uav_swarm_control.models import NeighborEncoderSpec, SharedActorCentralCritic


def _teacher_evidence(tmp_path: Path) -> tuple[Path, Path]:
    task = load_communication_experiment_config("configs/experiment/stage11_plane_4uav.yaml")
    model = SharedActorCentralCritic(
        local_observation_size=40,
        centralized_state_size=80,
        action_size=3,
        num_agents=4,
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(16,),
        initial_log_standard_deviation=-0.5,
        neighbor_encoder=NeighborEncoderSpec(6, 3, 6, 8, (8,)),
    )
    checkpoint = tmp_path / "model.pt"
    configuration = json.loads(json.dumps(asdict(task), sort_keys=True))
    save_mappo_checkpoint(
        checkpoint,
        model,
        metadata={
            "manifest_fingerprint": "manifest",
            "training_seed": 11,
            "configuration": configuration,
        },
    )
    result = tmp_path / "result.json"
    conditions = {
        name: {
            "collision_free_success": 0.9,
            "collision_any": 0.02,
            "mean_normalized_shape_rmse": 0.1,
        }
        for name in ("k2-global-mixed", "kall-global-mixed")
    }
    save_json_artifact(
        result,
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "manifest_fingerprint": "manifest",
            "training_seed": 11,
            "condition_summaries": conditions,
        },
    )
    return checkpoint, result


def test_validated_teacher_binds_checkpoint_configuration_and_metrics(tmp_path: Path) -> None:
    checkpoint, result = _teacher_evidence(tmp_path)
    task = load_communication_experiment_config("configs/experiment/stage11_plane_4uav.yaml")
    study = load_deployment_study_config("configs/deployment/stage12_policy_compression.yaml")
    _, validation = validate_teacher(checkpoint, result, task, study)
    assert validation["scientific_valid"] is True
    assert validation["behavioral_gate_passed"] is True


def test_teacher_gate_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    checkpoint, result = _teacher_evidence(tmp_path)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    task = load_communication_experiment_config("configs/experiment/stage11_plane_4uav.yaml")
    study = load_deployment_study_config("configs/deployment/stage12_policy_compression.yaml")
    with pytest.raises(ValueError, match="checksum"):
        validate_teacher(checkpoint, result, task, study)
