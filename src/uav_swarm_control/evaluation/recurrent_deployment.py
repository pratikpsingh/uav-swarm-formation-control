"""Compression and actor-only export for paper-aligned recurrent policies."""

import copy
import hashlib
import json
import logging
import math
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from uav_swarm_control.actions import DirectionSpeedActions, direction_speed_to_normalized_velocity
from uav_swarm_control.algorithms.mappo import load_recurrent_mappo_checkpoint
from uav_swarm_control.communication import CommunicationRegimen, CommunicationRegimenKind
from uav_swarm_control.configuration.deployment import DeploymentStudyConfig
from uav_swarm_control.configuration.recurrent_study import RecurrentStudyConfig
from uav_swarm_control.controllers.contracts import Controller, StateAwareController
from uav_swarm_control.deployment.artifacts import (
    benchmark_exported_actor,
    export_actor,
    quantize_int8_dynamic,
)
from uav_swarm_control.deployment.contracts import (
    DeploymentActionProfile,
    DeploymentArchitecture,
)
from uav_swarm_control.deployment.distillation import TeacherTrajectory, distill_student
from uav_swarm_control.deployment.models import DeploymentController
from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.benchmark import (
    average_episodes,
    evaluate_benchmark,
    summarize_seeds,
)
from uav_swarm_control.evaluation.communication import student_t_critical_95
from uav_swarm_control.evaluation.provenance import (
    collect_provenance,
    stable_digest,
    write_source_snapshot,
)
from uav_swarm_control.evaluation.recurrent_controller import RecurrentActorController
from uav_swarm_control.models.neighbor_encoder import masked_mean_neighbor_features
from uav_swarm_control.models.recurrent_actor_critic import (
    PaperRecurrentActorCritic,
)
from uav_swarm_control.observations import encode_local_observations
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed

LOGGER = logging.getLogger(__name__)
_CONTEXT_PREFIX = "context/"


class RecurrentTeacherActor(nn.Module):
    """Actor-only batch-first graph with caller-owned LSTM state."""

    def __init__(self, model: PaperRecurrentActorCritic) -> None:
        super().__init__()
        self.local_observation_size = model.local_observation_size
        self.action_size = model.action_size
        self.width = model.recurrent_hidden_size
        self.recurrent_layers = model.recurrent_layers
        self.neighbor_encoder = model.neighbor_encoder
        self.neighbor_embedding = copy.deepcopy(model.neighbor_embedding)
        self.actor_input = copy.deepcopy(model.actor_input)
        self.actor_input_norm = copy.deepcopy(model.actor_input_norm)
        self.actor_lstm = copy.deepcopy(model.actor_lstm)
        self.actor_output_norm = copy.deepcopy(model.actor_output_norm)
        self.actor_output = copy.deepcopy(model.actor_output)

    def forward(
        self,
        observations: Tensor,
        hidden: Tensor,
        cell: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != self.local_observation_size:
            raise ValueError("observations must have shape [batch, time, local_features].")
        time_major = observations.permute(1, 0, 2)
        features = time_major
        if self.neighbor_encoder is not None:
            if self.neighbor_embedding is None:
                raise RuntimeError("neighbor encoder graph is incomplete.")
            features = masked_mean_neighbor_features(
                time_major,
                self.neighbor_encoder,
                self.neighbor_embedding,
            )
        features = self.actor_input_norm(torch.relu(self.actor_input(features)))
        output, (next_hidden, next_cell) = self.actor_lstm(features, (hidden, cell))
        actions = torch.tanh(self.actor_output(self.actor_output_norm(output)))
        return actions.permute(1, 0, 2), next_hidden, next_cell


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return cast(dict[str, object], value)


def _float(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a numeric metric.")
    return float(value)


def validate_recurrent_teacher(
    checkpoint: Path,
    result_path: Path,
    task: RecurrentStudyConfig,
    study: DeploymentStudyConfig,
) -> tuple[PaperRecurrentActorCritic, dict[str, object]]:
    """Bind recurrent weights, study configuration, result checksum, and behavior gates."""
    result = _mapping(json.loads(result_path.read_text(encoding="utf-8")), "teacher result")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if result.get("checkpoint_sha256") != digest:
        raise ValueError("teacher result checksum does not match the supplied checkpoint.")
    model, metadata = load_recurrent_mappo_checkpoint(checkpoint, device="cpu")
    if model.neighbor_encoder is None:
        raise ValueError("recurrent compression requires a masked-set teacher.")
    if model.action_size != 4:
        raise ValueError("recurrent compression requires four direction-speed actor outputs.")
    if metadata.get("manifest_fingerprint") != result.get("manifest_fingerprint"):
        raise ValueError("teacher checkpoint and result have different manifest fingerprints.")
    if metadata.get("training_seed") != result.get("training_seed"):
        raise ValueError("teacher checkpoint and result have different training seeds.")
    expected = json.loads(json.dumps(asdict(task), sort_keys=True))
    if metadata.get("configuration") != expected:
        raise ValueError("teacher checkpoint configuration does not match the requested task.")

    condition_summaries = _mapping(result.get("condition_summaries"), "condition summaries")
    checks: list[dict[str, object]] = []
    passed = True
    for gate in study.teacher_gates:
        metrics = _mapping(condition_summaries.get(gate.condition), gate.condition)
        values = {
            "collision_free_success": _float(metrics, "collision_free_success"),
            "collision_any": _float(metrics, "collision_any"),
            "mean_normalized_shape_rmse": _float(metrics, "mean_normalized_shape_rmse"),
        }
        condition_passed = (
            values["collision_free_success"] >= gate.minimum_collision_free_success
            and values["collision_any"] <= gate.maximum_collision_any
            and values["mean_normalized_shape_rmse"] <= gate.maximum_mean_normalized_shape_rmse
        )
        passed &= condition_passed
        checks.append(
            {
                "condition": gate.condition,
                "metrics": values,
                "gate": asdict(gate),
                "passed": condition_passed,
            }
        )
    scientific = task.communication.profile == "research" and study.profile == "research" and passed
    if study.profile == "research" and not scientific:
        raise ValueError("recurrent teacher failed the configured research gates.")
    return model.eval(), {
        "checkpoint_sha256": digest,
        "result_path": str(result_path.resolve()),
        "training_seed": result["training_seed"],
        "manifest_fingerprint": result["manifest_fingerprint"],
        "behavioral_gate_passed": passed,
        "scientific_valid": scientific,
        "checks": checks,
        "label": "validated-research-teacher" if scientific else "plumbing-only-smoke-teacher",
    }


def _factory(task: RecurrentStudyConfig, condition_index: int) -> CommunicationFormationEnvironment:
    communication = task.communication
    condition = communication.evaluation_conditions[condition_index]
    regimen = CommunicationRegimen(
        f"deploy-{condition.name}",
        CommunicationRegimenKind.FIXED,
        (condition,),
    )
    return CommunicationFormationEnvironment(
        communication.physics,
        communication.pose,
        communication.variant,
        communication.field,
        regimen,
        payload_bytes_per_neighbor=communication.encoder.payload_bytes_per_neighbor,
    )


def collect_recurrent_teacher_trajectories(
    task: RecurrentStudyConfig,
    teacher: PaperRecurrentActorCritic,
    *,
    episodes_per_condition: int,
    seed: int,
) -> tuple[TeacherTrajectory, ...]:
    """Collect bounded actor outputs while retaining complete per-agent sequences."""
    trajectories: list[TeacherTrajectory] = []
    communication = task.communication
    for condition_index, condition in enumerate(communication.evaluation_conditions):
        environment = _factory(task, condition_index)
        try:
            for episode in range(episodes_per_condition):
                episode_seed = derive_indexed_seed(
                    seed,
                    RandomStream.EVALUATION,
                    condition_index,
                    episode,
                )
                observations = environment.reset(seed=episode_seed).observations
                state = teacher.initial_actor_state(
                    len(environment.agent_ids), device=torch.device("cpu")
                )
                keep = torch.zeros(len(environment.agent_ids), dtype=torch.bool)
                observation_rows: list[Tensor] = []
                action_rows: list[Tensor] = []
                for _ in range(communication.mappo.experiment.environment.max_episode_steps):
                    encoded = torch.as_tensor(
                        np.array(encode_local_observations(observations), copy=True),
                        dtype=torch.float32,
                    )
                    with torch.no_grad():
                        bounded, state = teacher.deterministic_actions(
                            encoded.unsqueeze(0),
                            state,
                            keep.unsqueeze(0),
                        )
                    actions = bounded.squeeze(0)
                    observation_rows.append(encoded)
                    action_rows.append(actions)
                    transition = environment.step(
                        direction_speed_to_normalized_velocity(
                            DirectionSpeedActions(
                                observations.agent_ids,
                                actions.numpy().astype(np.float32, copy=False),
                            ),
                            direction_epsilon=task.recurrent.direction_epsilon,
                        )
                    )
                    observations = transition.observations
                    keep = torch.ones_like(keep)
                    if transition.episode_done:
                        break
                else:
                    raise RuntimeError("teacher collection exceeded the configured horizon.")
                trajectories.append(
                    TeacherTrajectory(
                        torch.stack(observation_rows),
                        torch.stack(action_rows),
                        condition.name,
                        episode_seed,
                    )
                )
        finally:
            environment.close()
    return tuple(trajectories)


def _task_summary(episodes: list[dict[str, float]]) -> dict[str, float]:
    return average_episodes(
        [
            {name: value for name, value in episode.items() if not name.startswith(_CONTEXT_PREFIX)}
            for episode in episodes
        ]
    )


def _evaluate_controller(
    task: RecurrentStudyConfig,
    controller: Controller | StateAwareController,
    episodes: int,
) -> dict[str, dict[str, float]]:
    communication = task.communication
    summaries: dict[str, dict[str, float]] = {}
    for index, condition in enumerate(communication.evaluation_conditions):
        results = evaluate_benchmark(
            lambda active=index: _factory(task, active),
            controller,
            episodes=episodes,
            seed=communication.evaluation_seed,
            horizon=communication.mappo.experiment.environment.max_episode_steps,
            time_step_seconds=communication.mappo.experiment.environment.time_step_seconds,
            collision_distance_m=communication.mappo.experiment.task.collision_distance_m,
            episode_metadata=lambda environment: (
                cast(CommunicationFormationEnvironment, environment).episode_metadata
            ),
            episode_metadata_prefix="context",
        )
        summaries[condition.name] = _task_summary(results)
    return summaries


def _summary_with_ci(records: list[dict[str, float]]) -> dict[str, object]:
    summary = summarize_seeds(records)
    count = len(records)
    critical = student_t_critical_95(count)
    metrics = cast(dict[str, dict[str, float]], summary["metrics"])
    for values in metrics.values():
        margin = critical * values["sample_std"] / math.sqrt(count)
        values["ci95_low"] = values["mean"] - margin
        values["ci95_high"] = values["mean"] + margin
    summary["confidence_interval"] = {
        "level": 0.95,
        "method": "two-sided Student-t across distillation seeds",
        "critical_value": critical,
    }
    return summary


def _fingerprint_trajectories(trajectories: tuple[TeacherTrajectory, ...]) -> str:
    digest = hashlib.sha256()
    for trajectory in trajectories:
        digest.update(trajectory.condition.encode())
        digest.update(str(trajectory.episode_seed).encode())
        digest.update(trajectory.observations.numpy().tobytes())
        digest.update(trajectory.actions.numpy().tobytes())
    return digest.hexdigest()


def run_recurrent_deployment_study(
    task: RecurrentStudyConfig,
    study: DeploymentStudyConfig,
    checkpoint: Path,
    teacher_result: Path,
    output: Path,
    *,
    project_root: Path,
) -> Path:
    """Distill, export, and evaluate compact direction-speed actors."""
    communication = task.communication
    if study.dataset_seed == communication.evaluation_seed:
        raise ValueError("distillation and closed-loop evaluation need different root seeds.")
    if (
        any(candidate.quantization == "int8-dynamic" for candidate in study.candidates)
        and find_spec("torchao") is None
    ):
        raise RuntimeError("INT8 candidates require uv sync --extra deployment.")
    teacher, validation = validate_recurrent_teacher(
        checkpoint,
        teacher_result,
        task,
        study,
    )
    directory = output / "recurrent" / study.profile / study.name
    directory.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "recurrent-policy-distillation",
        "task_configuration": asdict(task),
        "deployment_configuration": asdict(study),
        "teacher_validation": validation,
        "provenance": collect_provenance(project_root),
        "measurement_boundaries": {
            "deployment_unit": "one deterministic local actor; critic and log-std excluded",
            "action_profile": DeploymentActionProfile.DIRECTION_SPEED.value,
            "flash_proxy": "serialized portable actor graph bytes",
            "recurrent_state": "logical FP32 hidden/cell bytes per UAV",
            "ram": "host process RSS is not target peak RAM",
            "latency": "host CPU latency is not flight-controller latency",
            "energy": "external target measurement only",
        },
    }
    manifest = cast(dict[str, object], json.loads(json.dumps(manifest, sort_keys=True)))
    manifest["fingerprint"] = stable_digest(manifest)
    save_json_artifact(directory / "manifest.json", manifest)
    write_source_snapshot(project_root, directory / "source.zip")

    trajectories = collect_recurrent_teacher_trajectories(
        task,
        teacher,
        episodes_per_condition=study.distillation.dataset_episodes_per_condition,
        seed=study.dataset_seed,
    )
    save_json_artifact(
        directory / "dataset.json",
        {
            "episode_count": len(trajectories),
            "agent_sequence_count": len(trajectories)
            * communication.mappo.experiment.formation.num_agents,
            "step_count": sum(item.observations.shape[0] for item in trajectories),
            "sha256": _fingerprint_trajectories(trajectories),
            "split_unit": "complete episode",
        },
    )

    teacher_actor = RecurrentTeacherActor(teacher).eval()
    teacher_directory = directory / "teacher"
    teacher_artifact, teacher_metadata = export_actor(
        teacher_actor,
        teacher_directory,
        architecture=DeploymentArchitecture.LSTM,
        observation_size=teacher.local_observation_size,
        action_size=teacher.action_size,
        recurrent_width=teacher.recurrent_hidden_size,
        candidate="teacher",
        quantization="none",
        structured_zero_channel_fraction=0.0,
        action_profile=DeploymentActionProfile.DIRECTION_SPEED,
    )
    teacher_benchmark = benchmark_exported_actor(
        teacher_artifact,
        teacher_directory / "artifact.json",
        warmup_iterations=study.benchmark.warmup_iterations,
        measured_iterations=study.benchmark.measured_iterations,
        torch_threads=study.benchmark.torch_threads,
    )
    teacher_task = _evaluate_controller(
        task,
        RecurrentActorController(
            teacher,
            direction_epsilon=task.recurrent.direction_epsilon,
        ),
        study.evaluation_episodes_per_condition,
    )
    save_json_artifact(
        teacher_directory / "result.json",
        {
            "artifact": teacher_metadata,
            "host_benchmark": teacher_benchmark,
            "task_conditions": teacher_task,
        },
    )

    teacher_spec = teacher.neighbor_encoder
    if teacher_spec is None:
        raise RuntimeError("validated recurrent teacher lost its neighbor specification.")
    all_results: dict[str, object] = {}
    for candidate in study.candidates:
        LOGGER.info("Recurrent compression candidate: %s", candidate.name)
        seed_results: list[dict[str, object]] = []
        for seed in study.distillation_seeds:
            trained = distill_student(
                candidate,
                trajectories,
                study.distillation,
                seed=seed,
                local_observation_size=teacher.local_observation_size,
                action_size=teacher.action_size,
                teacher_neighbor_spec=teacher_spec,
            )
            deployed = (
                quantize_int8_dynamic(trained.model)
                if candidate.quantization == "int8-dynamic"
                else trained.model
            )
            seed_directory = directory / candidate.name / f"seed-{seed}"
            artifact, artifact_metadata = export_actor(
                deployed,
                seed_directory,
                architecture=candidate.architecture,
                observation_size=teacher.local_observation_size,
                action_size=teacher.action_size,
                recurrent_width=candidate.width,
                candidate=candidate.name,
                quantization=candidate.quantization,
                structured_zero_channel_fraction=trained.structured_zero_channel_fraction,
                action_profile=DeploymentActionProfile.DIRECTION_SPEED,
            )
            host = benchmark_exported_actor(
                artifact,
                seed_directory / "artifact.json",
                warmup_iterations=study.benchmark.warmup_iterations,
                measured_iterations=study.benchmark.measured_iterations,
                torch_threads=study.benchmark.torch_threads,
            )
            task_conditions = _evaluate_controller(
                task,
                DeploymentController(
                    deployed,
                    candidate.architecture,
                    action_profile=DeploymentActionProfile.DIRECTION_SPEED,
                    direction_epsilon=task.recurrent.direction_epsilon,
                ),
                study.evaluation_episodes_per_condition,
            )
            record: dict[str, object] = {
                "distillation_seed": seed,
                "distillation": {
                    "training_mse": trained.training_mse,
                    "validation_mse": trained.validation_mse,
                    "training_trajectories": trained.training_trajectories,
                    "validation_trajectories": trained.validation_trajectories,
                },
                "artifact": artifact_metadata,
                "host_benchmark": host,
                "energy": {
                    "status": study.energy.status,
                    "reason": study.energy.reason,
                    "joules_per_inference": study.energy.joules_per_inference.get(candidate.name),
                },
                "task_conditions": task_conditions,
            }
            save_json_artifact(seed_directory / "result.json", record)
            seed_results.append(record)

        systems = [
            {
                "parameter_count": _float(
                    cast(dict[str, object], item["artifact"]), "parameter_count"
                ),
                "logical_parameter_bytes": _float(
                    cast(dict[str, object], item["artifact"]),
                    "logical_parameter_bytes",
                ),
                "logical_recurrent_state_bytes_per_agent": _float(
                    cast(dict[str, object], item["artifact"]),
                    "logical_recurrent_state_bytes_per_agent",
                ),
                "artifact_bytes": _float(
                    cast(dict[str, object], item["artifact"]), "artifact_bytes"
                ),
                "latency_ms_p50": _float(
                    cast(dict[str, object], item["host_benchmark"]),
                    "latency_ms_p50",
                ),
                "latency_ms_p95": _float(
                    cast(dict[str, object], item["host_benchmark"]),
                    "latency_ms_p95",
                ),
                "peak_process_rss_bytes": _float(
                    cast(dict[str, object], item["host_benchmark"]),
                    "peak_process_rss_bytes",
                ),
                "validation_mse": _float(
                    cast(dict[str, object], item["distillation"]),
                    "validation_mse",
                ),
            }
            for item in seed_results
        ]
        conditions = {
            condition.name: _summary_with_ci(
                [
                    cast(dict[str, dict[str, float]], item["task_conditions"])[condition.name]
                    for item in seed_results
                ]
            )
            for condition in communication.evaluation_conditions
        }
        candidate_summary = {
            "candidate": asdict(candidate),
            "systems_and_imitation": _summary_with_ci(systems),
            "task_conditions": conditions,
        }
        all_results[candidate.name] = candidate_summary
        save_json_artifact(directory / candidate.name / "summary.json", candidate_summary)

    return save_json_artifact(
        directory / "summary.json",
        {
            "artifact_schema_version": 1,
            "manifest_fingerprint": manifest["fingerprint"],
            "scientific_valid": validation["scientific_valid"],
            "teacher": {
                "artifact": teacher_metadata,
                "host_benchmark": teacher_benchmark,
                "task_conditions": teacher_task,
            },
            "candidates": all_results,
            "selection_rule": (
                "Reject candidates that violate task gates, then compare target-measured "
                "flash, peak RAM, latency, and energy. Host measurements are screening only."
            ),
        },
    )


__all__ = [
    "RecurrentTeacherActor",
    "collect_recurrent_teacher_trajectories",
    "run_recurrent_deployment_study",
    "validate_recurrent_teacher",
]
