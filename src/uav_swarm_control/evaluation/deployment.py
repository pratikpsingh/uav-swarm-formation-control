"""Validated-teacher compression study with closed-loop and systems measurements."""

import hashlib
import json
import logging
import math
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from statistics import NormalDist
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from uav_swarm_control.algorithms.mappo import load_mappo_checkpoint
from uav_swarm_control.communication import CommunicationRegimen, CommunicationRegimenKind
from uav_swarm_control.configuration.communication import CommunicationExperimentConfig
from uav_swarm_control.configuration.deployment import DeploymentStudyConfig
from uav_swarm_control.controllers.contracts import Controller, StateAwareController
from uav_swarm_control.deployment.artifacts import (
    benchmark_exported_actor,
    export_actor,
    quantize_int8_dynamic,
)
from uav_swarm_control.deployment.distillation import TeacherTrajectory, distill_student
from uav_swarm_control.deployment.models import DeploymentArchitecture, DeploymentController
from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.benchmark import (
    ActorController,
    average_episodes,
    evaluate_benchmark,
    summarize_seeds,
)
from uav_swarm_control.evaluation.provenance import (
    collect_provenance,
    stable_digest,
    write_source_snapshot,
)
from uav_swarm_control.models import SharedActorCentralCritic
from uav_swarm_control.observations import encode_local_observations
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed

LOGGER = logging.getLogger(__name__)
_CONTEXT_PREFIX = "context/"


class TeacherActor(nn.Module):
    """Actor-only deterministic wrapper; critic and exploration variance are absent."""

    def __init__(self, model: SharedActorCentralCritic) -> None:
        super().__init__()
        self.actor = model.actor

    def forward(self, observations: Tensor) -> Tensor:
        return torch.tanh(self.actor(observations))


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return cast(dict[str, object], value)


def _float(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a numeric metric.")
    return float(value)


def validate_teacher(
    checkpoint: Path,
    result_path: Path,
    task: CommunicationExperimentConfig,
    study: DeploymentStudyConfig,
) -> tuple[SharedActorCentralCritic, dict[str, object]]:
    """Bind checkpoint bytes, training metadata, task configuration, and metric gates."""
    result = _mapping(json.loads(result_path.read_text(encoding="utf-8")), "teacher result")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if result.get("checkpoint_sha256") != digest:
        raise ValueError("teacher result checksum does not match the supplied checkpoint.")
    model, metadata = load_mappo_checkpoint(checkpoint, device="cpu")
    if model.neighbor_encoder is None:
        raise ValueError("Stage 12 requires a Stage 11 masked-neighbor teacher.")
    if metadata.get("manifest_fingerprint") != result.get("manifest_fingerprint"):
        raise ValueError("teacher checkpoint and result have different manifest fingerprints.")
    if metadata.get("training_seed") != result.get("training_seed"):
        raise ValueError("teacher checkpoint and result have different training seeds.")
    saved_configuration = metadata.get("configuration")
    expected_configuration = json.loads(json.dumps(asdict(task), sort_keys=True))
    if saved_configuration != expected_configuration:
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
    scientific = task.profile == "research" and study.profile == "research" and passed
    if study.profile == "research" and not scientific:
        raise ValueError("teacher failed the research profile or configured behavioral gates.")
    validation = {
        "checkpoint_sha256": digest,
        "result_path": str(result_path.resolve()),
        "training_seed": result["training_seed"],
        "manifest_fingerprint": result["manifest_fingerprint"],
        "behavioral_gate_passed": passed,
        "scientific_valid": scientific,
        "checks": checks,
        "label": "validated-research-teacher" if scientific else "plumbing-only-smoke-teacher",
    }
    return model.eval(), validation


def _factory(
    task: CommunicationExperimentConfig, condition_index: int
) -> CommunicationFormationEnvironment:
    condition = task.evaluation_conditions[condition_index]
    regimen = CommunicationRegimen(
        f"deploy-{condition.name}", CommunicationRegimenKind.FIXED, (condition,)
    )
    return CommunicationFormationEnvironment(
        task.physics,
        task.pose,
        task.variant,
        task.field,
        regimen,
        payload_bytes_per_neighbor=task.encoder.payload_bytes_per_neighbor,
    )


def collect_teacher_trajectories(
    task: CommunicationExperimentConfig,
    teacher: SharedActorCentralCritic,
    *,
    episodes_per_condition: int,
    seed: int,
) -> tuple[TeacherTrajectory, ...]:
    """Collect complete teacher episodes; the later split never divides an episode."""
    trajectories: list[TeacherTrajectory] = []
    for condition_index, condition in enumerate(task.evaluation_conditions):
        environment = _factory(task, condition_index)
        try:
            for episode in range(episodes_per_condition):
                episode_seed = derive_indexed_seed(
                    seed, RandomStream.EVALUATION, condition_index, episode
                )
                reset = environment.reset(seed=episode_seed)
                observations = reset.observations
                observation_rows: list[Tensor] = []
                action_rows: list[Tensor] = []
                for _ in range(task.mappo.experiment.environment.max_episode_steps):
                    encoded = torch.as_tensor(
                        np.array(encode_local_observations(observations), copy=True),
                        dtype=torch.float32,
                    )
                    with torch.no_grad():
                        actions = teacher.deterministic_actions(encoded)
                    observation_rows.append(encoded)
                    action_rows.append(actions)
                    transition = environment.step(
                        NormalizedVelocityActions(
                            observations.agent_ids, actions.numpy().astype(np.float32)
                        )
                    )
                    observations = transition.observations
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
    task: CommunicationExperimentConfig,
    controller: Controller | StateAwareController,
    episodes: int,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for index, condition in enumerate(task.evaluation_conditions):
        results = evaluate_benchmark(
            lambda active=index: _factory(task, active),
            controller,
            episodes=episodes,
            seed=task.evaluation_seed,
            horizon=task.mappo.experiment.environment.max_episode_steps,
            time_step_seconds=task.mappo.experiment.environment.time_step_seconds,
            collision_distance_m=task.mappo.experiment.task.collision_distance_m,
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
    degrees = count - 1
    if degrees == 4:
        critical = 2.7764451051977987
    else:
        z = NormalDist().inv_cdf(0.975)
        inverse = 1.0 / degrees
        critical = z + (z**3 + z) * inverse / 4.0
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


def run_deployment_study(
    task: CommunicationExperimentConfig,
    study: DeploymentStudyConfig,
    checkpoint: Path,
    teacher_result: Path,
    output: Path,
    *,
    project_root: Path,
) -> Path:
    """Distill, compress, export, and measure every configured actor treatment."""
    if study.dataset_seed == task.evaluation_seed:
        raise ValueError("distillation and closed-loop evaluation must use different root seeds.")
    if (
        any(candidate.quantization == "int8-dynamic" for candidate in study.candidates)
        and find_spec("torchao") is None
    ):
        raise RuntimeError("INT8 candidates require `uv sync --extra deployment`.")
    teacher, validation = validate_teacher(checkpoint, teacher_result, task, study)
    directory = output / study.profile / study.name
    directory.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "stage12-validated-teacher-policy-compression",
        "task_configuration": asdict(task),
        "deployment_configuration": asdict(study),
        "teacher_validation": validation,
        "provenance": collect_provenance(project_root),
        "measurement_boundaries": {
            "deployment_unit": "one deterministic decentralized actor; critic excluded",
            "flash_proxy": "serialized portable actor graph bytes",
            "ram": "fresh host Python process peak RSS, not embedded target RAM",
            "latency": "single-agent host CPU inference, not flight-controller latency",
            "energy": "external measurement only; never estimated from host timing",
        },
    }
    manifest = cast(dict[str, object], json.loads(json.dumps(manifest, sort_keys=True)))
    manifest["fingerprint"] = stable_digest(manifest)
    save_json_artifact(directory / "manifest.json", manifest)
    write_source_snapshot(project_root, directory / "source.zip")

    trajectories = collect_teacher_trajectories(
        task,
        teacher,
        episodes_per_condition=study.distillation.dataset_episodes_per_condition,
        seed=study.dataset_seed,
    )
    save_json_artifact(
        directory / "dataset.json",
        {
            "episode_count": len(trajectories),
            "agent_sequence_count": len(trajectories) * teacher.num_agents,
            "step_count": sum(item.observations.shape[0] for item in trajectories),
            "sha256": _fingerprint_trajectories(trajectories),
            "split_unit": "complete episode",
            "raw_storage": "not persisted; deterministically regenerated from teacher and manifest",
        },
    )

    teacher_neighbor_spec = teacher.neighbor_encoder
    if teacher_neighbor_spec is None:
        raise RuntimeError("validated teacher unexpectedly lost its neighbor specification.")
    teacher_actor = TeacherActor(teacher).eval()
    teacher_directory = directory / "teacher"
    teacher_artifact, teacher_metadata = export_actor(
        teacher_actor,
        teacher_directory,
        architecture=DeploymentArchitecture.FEED_FORWARD,
        observation_size=teacher.local_observation_size,
        action_size=teacher.action_size,
        recurrent_width=0,
        candidate="teacher",
        quantization="none",
        structured_zero_channel_fraction=0.0,
    )
    teacher_benchmark = benchmark_exported_actor(
        teacher_artifact,
        teacher_directory / "artifact.json",
        warmup_iterations=study.benchmark.warmup_iterations,
        measured_iterations=study.benchmark.measured_iterations,
        torch_threads=study.benchmark.torch_threads,
    )
    teacher_task = _evaluate_controller(
        task, ActorController(teacher), study.evaluation_episodes_per_condition
    )
    save_json_artifact(
        teacher_directory / "result.json",
        {
            "artifact": teacher_metadata,
            "host_benchmark": teacher_benchmark,
            "task_conditions": teacher_task,
        },
    )

    all_results: dict[str, object] = {}
    for candidate in study.candidates:
        LOGGER.info("Compression candidate: %s", candidate.name)
        seed_results: list[dict[str, object]] = []
        for seed in study.distillation_seeds:
            trained = distill_student(
                candidate,
                trajectories,
                study.distillation,
                seed=seed,
                local_observation_size=teacher.local_observation_size,
                action_size=teacher.action_size,
                teacher_neighbor_spec=teacher_neighbor_spec,
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
            )
            host = benchmark_exported_actor(
                artifact,
                seed_directory / "artifact.json",
                warmup_iterations=study.benchmark.warmup_iterations,
                measured_iterations=study.benchmark.measured_iterations,
                torch_threads=study.benchmark.torch_threads,
            )
            controller = DeploymentController(deployed, candidate.architecture)
            task_conditions = _evaluate_controller(
                task, controller, study.evaluation_episodes_per_condition
            )
            energy_value = study.energy.joules_per_inference.get(candidate.name)
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
                    "joules_per_inference": energy_value,
                },
                "task_conditions": task_conditions,
            }
            save_json_artifact(seed_directory / "result.json", record)
            seed_results.append(record)
        systems_records = [
            {
                "parameter_count": _float(
                    cast(dict[str, object], item["artifact"]), "parameter_count"
                ),
                "artifact_bytes": _float(
                    cast(dict[str, object], item["artifact"]), "artifact_bytes"
                ),
                "latency_ms_p50": _float(
                    cast(dict[str, object], item["host_benchmark"]), "latency_ms_p50"
                ),
                "latency_ms_p95": _float(
                    cast(dict[str, object], item["host_benchmark"]), "latency_ms_p95"
                ),
                "peak_process_rss_bytes": _float(
                    cast(dict[str, object], item["host_benchmark"]),
                    "peak_process_rss_bytes",
                ),
                "validation_mse": _float(
                    cast(dict[str, object], item["distillation"]), "validation_mse"
                ),
            }
            for item in seed_results
        ]
        condition_summaries = {
            condition.name: _summary_with_ci(
                [
                    cast(dict[str, dict[str, float]], item["task_conditions"])[condition.name]
                    for item in seed_results
                ]
            )
            for condition in task.evaluation_conditions
        }
        candidate_summary: dict[str, object] = {
            "candidate": asdict(candidate),
            "systems_and_imitation": _summary_with_ci(systems_records),
            "task_conditions": condition_summaries,
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
                "Choose only from candidates meeting predeclared task gates; among those, compare "
                "target-measured flash, RAM, latency, and energy. Host measurements are "
                "screening data."
            ),
        },
    )


__all__ = [
    "TeacherActor",
    "collect_teacher_trajectories",
    "run_deployment_study",
    "validate_teacher",
]
