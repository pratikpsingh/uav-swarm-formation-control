"""Auditable recurrent training and evaluation across research treatments."""

import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import cast

from uav_swarm_control.algorithms.mappo.recurrent_checkpoint import (
    load_recurrent_mappo_checkpoint,
    save_recurrent_mappo_checkpoint,
)
from uav_swarm_control.algorithms.mappo.recurrent_trainer import (
    RecurrentMAPPOUpdateMetrics,
    train_recurrent_mappo,
)
from uav_swarm_control.communication import (
    CommunicationCondition,
    CommunicationRegimen,
    CommunicationRegimenKind,
)
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.configuration.recurrent_mappo import RecurrentObservationProfile
from uav_swarm_control.configuration.recurrent_study import RecurrentStudyConfig
from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
from uav_swarm_control.environments.morphing import MorphingFormationEnvironment
from uav_swarm_control.environments.research import (
    MissionFormationEnvironment,
    PooledFormationEnvironment,
)
from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.benchmark import summarize_seeds
from uav_swarm_control.evaluation.provenance import (
    collect_provenance,
    stable_digest,
    write_source_snapshot,
)
from uav_swarm_control.evaluation.recurrent_benchmark import (
    ActionTransform,
    evaluate_recurrent_environment,
    numeric_mean,
    velocity_disturbance,
)
from uav_swarm_control.evaluation.recurrent_controller import RecurrentActorController
from uav_swarm_control.models.recurrent_actor_critic import PaperRecurrentActorCritic
from uav_swarm_control.obstacles import ObstacleScenario

LOGGER = logging.getLogger(__name__)
type PhysicalEnvironment = (
    CommunicationFormationEnvironment
    | PooledFormationEnvironment
    | MissionFormationEnvironment
    | MorphingFormationEnvironment
)
type PhysicalFactory = Callable[[], PhysicalEnvironment]


class RecurrentTreatment(StrEnum):
    BASE = "base"
    POOLED_FORMATIONS = "pooled-formations"
    MISSION = "mission"
    OBSTACLES = "obstacles"
    NEIGHBORS = "neighbors"
    RECOVERY = "recovery"
    MORPHING = "morphing"


def recurrent_study_smoke_config(config: RecurrentStudyConfig) -> RecurrentStudyConfig:
    """Bound compute while retaining all configured factors and training seeds."""
    communication = config.communication
    experiment = replace(
        communication.mappo.experiment,
        environment=replace(communication.mappo.experiment.environment, max_episode_steps=48),
    )
    algorithm = replace(
        communication.mappo.algorithm,
        ppo=replace(
            communication.mappo.algorithm.ppo,
            total_steps=128,
            rollout_steps=32,
            minibatch_size=32,
            update_epochs=1,
            device="cpu",
        ),
        num_environments=1,
    )
    communication = replace(
        communication,
        mappo=replace(
            communication.mappo,
            experiment=experiment,
            algorithm=algorithm,
            evaluation_episodes=2,
        ),
        physics=PyBulletExperimentConfig(experiment, communication.physics.simulator),
        profile="smoke",
    )
    recurrent = replace(
        config.recurrent,
        mappo=algorithm,
        sequence_length=min(config.recurrent.sequence_length, 8),
        sequences_per_minibatch=1,
    )
    recovery = replace(
        config.recovery,
        disturbance_step=min(
            config.recovery.disturbance_step,
            experiment.environment.max_episode_steps / 3,
        ),
    )
    return replace(
        config,
        communication=communication,
        recurrent=recurrent,
        recovery=recovery,
    )


def _find_condition(
    config: RecurrentStudyConfig,
    *,
    neighbors: int,
    scenario: ObstacleScenario,
) -> CommunicationCondition:
    candidates = [
        condition
        for condition in config.communication.evaluation_conditions
        if condition.requested_neighbors == neighbors
        and condition.obstacle_scenario is scenario
        and condition.sensing_radius_m is None
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one unlimited-range condition for k={neighbors}, obstacle={scenario.value}."
        )
    return candidates[0]


def _regimens(
    config: RecurrentStudyConfig,
    treatment: RecurrentTreatment,
) -> tuple[CommunicationRegimen, ...]:
    all_neighbors = config.communication.mappo.experiment.formation.num_agents - 1
    if treatment is RecurrentTreatment.OBSTACLES:
        return tuple(
            CommunicationRegimen(
                f"fixed-{scenario.value}",
                CommunicationRegimenKind.FIXED,
                (_find_condition(config, neighbors=all_neighbors, scenario=scenario),),
            )
            for scenario in ObstacleScenario
        )
    if treatment is RecurrentTreatment.NEIGHBORS:
        counts = sorted(
            {
                condition.requested_neighbors
                for condition in config.communication.evaluation_conditions
            }
        )
        fixed = tuple(
            CommunicationRegimen(
                f"fixed-k{count}",
                CommunicationRegimenKind.FIXED,
                (
                    _find_condition(
                        config,
                        neighbors=count,
                        scenario=ObstacleScenario.NONE,
                    ),
                ),
            )
            for count in counts
        )
        variable = next(
            regimen
            for regimen in config.communication.training_regimens
            if regimen.kind is CommunicationRegimenKind.VARIABLE
        )
        return (*fixed, variable)
    scenario = (
        ObstacleScenario.MIXED_DYNAMIC
        if treatment is RecurrentTreatment.MORPHING
        else ObstacleScenario.NONE
    )
    return (
        CommunicationRegimen(
            f"fixed-k{all_neighbors}-{scenario.value}",
            CommunicationRegimenKind.FIXED,
            (_find_condition(config, neighbors=all_neighbors, scenario=scenario),),
        ),
    )


def _conditions(
    config: RecurrentStudyConfig,
    treatment: RecurrentTreatment,
) -> tuple[CommunicationCondition, ...]:
    all_neighbors = config.communication.mappo.experiment.formation.num_agents - 1
    if treatment is RecurrentTreatment.OBSTACLES:
        return tuple(
            _find_condition(config, neighbors=all_neighbors, scenario=scenario)
            for scenario in ObstacleScenario
        )
    if treatment is RecurrentTreatment.NEIGHBORS:
        return config.communication.evaluation_conditions
    scenario = (
        ObstacleScenario.MIXED_DYNAMIC
        if treatment is RecurrentTreatment.MORPHING
        else ObstacleScenario.NONE
    )
    return (_find_condition(config, neighbors=all_neighbors, scenario=scenario),)


def _factory(
    config: RecurrentStudyConfig,
    treatment: RecurrentTreatment,
    physics: PyBulletExperimentConfig,
    regimen: CommunicationRegimen,
    *,
    training_steps: int | None,
) -> PhysicalFactory:
    pose = config.communication.pose
    variant = config.communication.variant
    field = config.communication.field
    payload = config.communication.encoder.payload_bytes_per_neighbor
    formations = config.formation_kinds
    if treatment in {
        RecurrentTreatment.BASE,
        RecurrentTreatment.OBSTACLES,
        RecurrentTreatment.NEIGHBORS,
        RecurrentTreatment.RECOVERY,
    }:

        def standard() -> CommunicationFormationEnvironment:
            return CommunicationFormationEnvironment(
                physics,
                pose,
                variant,
                field,
                regimen,
                payload_bytes_per_neighbor=payload,
                training_steps_per_environment=training_steps,
            )

        return standard
    if treatment is RecurrentTreatment.POOLED_FORMATIONS:

        def pooled() -> PooledFormationEnvironment:
            return PooledFormationEnvironment(
                physics,
                pose,
                variant,
                field,
                regimen,
                formation_kinds=formations,
                payload_bytes_per_neighbor=payload,
                training_steps_per_environment=training_steps,
            )

        return pooled
    if treatment is RecurrentTreatment.MISSION:
        mission = config.mission

        def mission_environment() -> MissionFormationEnvironment:
            return MissionFormationEnvironment(
                physics,
                pose,
                variant,
                field,
                regimen,
                formation_kinds=formations,
                payload_bytes_per_neighbor=payload,
                initial_ground_altitude_m=mission.initial_ground_altitude_m,
                construction_altitude_m=mission.construction_altitude_m,
                construction_hold_steps=mission.construction_hold_steps,
                waypoint_spacing_m=mission.waypoint_spacing_m,
                waypoint_hold_steps=mission.waypoint_hold_steps,
                training_steps_per_environment=training_steps,
            )

        return mission_environment
    morph = config.morphing

    def morphing_environment() -> MorphingFormationEnvironment:
        return MorphingFormationEnvironment(
            physics,
            pose,
            variant,
            field,
            regimen,
            formation_kinds=formations,
            payload_bytes_per_neighbor=payload,
            morph_rate_per_second=morph.rate_per_second,
            release_hold_steps=morph.release_hold_steps,
            trigger_clearance_m=morph.trigger_clearance_m,
            training_steps_per_environment=training_steps,
        )

    return morphing_environment


def run_recurrent_study(
    config: RecurrentStudyConfig,
    treatment: RecurrentTreatment,
    output: Path,
    *,
    project_root: Path,
    resume: bool = False,
) -> Path:
    """Train independent recurrent policies and evaluate paired held-out conditions."""
    if config.recurrent.observation_profile is not RecurrentObservationProfile.MASKED_SET:
        raise ValueError("physical studies require the masked-set observation profile.")
    experiment = config.communication.mappo.experiment
    directory = output / config.communication.profile / treatment.value / experiment.name
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "paper-recurrent-mappo",
        "treatment": treatment.value,
        "configuration": asdict(config),
        "provenance": collect_provenance(project_root),
        "information_boundary": {
            "actor": "local observation plus per-UAV memory",
            "critic": "centralized state during training only",
            "obstacles": "exact simulator state",
        },
    }
    manifest = cast(
        dict[str, object], json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    )
    manifest["fingerprint"] = stable_digest(manifest)
    manifest_path = directory / "manifest.json"
    if directory.exists():
        if not resume:
            raise FileExistsError(f"output exists: {directory}; use --resume or a new output.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("resume requires identical configuration, source, and runtime.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        save_json_artifact(manifest_path, manifest)
        write_source_snapshot(project_root, directory / "source.zip")
    summaries = {
        regimen.name: _run_regimen(
            config,
            treatment,
            regimen,
            directory / regimen.name,
            manifest["fingerprint"],
            resume=resume,
        )
        for regimen in _regimens(config, treatment)
    }
    return save_json_artifact(
        directory / "summary.json",
        {
            "artifact_schema_version": 1,
            "manifest_fingerprint": manifest["fingerprint"],
            "treatment": treatment.value,
            "training_seeds": list(config.communication.training_seeds),
            "evaluation_seed": config.communication.evaluation_seed,
            "regimens": summaries,
        },
    )


def _run_regimen(
    config: RecurrentStudyConfig,
    treatment: RecurrentTreatment,
    regimen: CommunicationRegimen,
    directory: Path,
    fingerprint: object,
    *,
    resume: bool,
) -> dict[str, object]:
    directory.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in config.communication.training_seeds:
        seed_directory = directory / f"seed-{seed}"
        result_path = seed_directory / "result.json"
        checkpoint = seed_directory / "model.pt"
        if seed_directory.exists():
            if not resume or not result_path.is_file() or not checkpoint.is_file():
                raise FileExistsError(f"incomplete seed directory: {seed_directory}")
            record = cast(dict[str, object], json.loads(result_path.read_text(encoding="utf-8")))
            if (
                record.get("manifest_fingerprint") != fingerprint
                or record.get("training_seed") != seed
                or record.get("checkpoint_sha256")
                != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ):
                raise ValueError(f"completed seed provenance mismatch: {seed_directory}")
        else:
            seed_directory.mkdir(exist_ok=False)
            record = _run_seed(config, treatment, regimen, seed, seed_directory, fingerprint)
            save_json_artifact(result_path, record)
        records.append(record)
    condition_summaries: dict[str, object] = {}
    for condition in _conditions(config, treatment):
        seed_means = [
            cast(dict[str, dict[str, float]], record["condition_summaries"])[condition.name]
            for record in records
        ]
        condition_summaries[condition.name] = summarize_seeds(seed_means)
    return {
        "regimen": asdict(regimen),
        "training": summarize_seeds(
            [cast(dict[str, float], record["training_summary"]) for record in records]
        ),
        "conditions": condition_summaries,
    }


def _run_seed(
    config: RecurrentStudyConfig,
    treatment: RecurrentTreatment,
    regimen: CommunicationRegimen,
    seed: int,
    directory: Path,
    fingerprint: object,
) -> dict[str, object]:
    experiment = replace(config.communication.mappo.experiment, seed=seed)
    physics = replace(config.communication.physics, experiment=experiment)
    training_steps = math.ceil(
        config.recurrent.mappo.ppo.total_steps / config.recurrent.mappo.num_environments
    )
    training_factory = _factory(config, treatment, physics, regimen, training_steps=training_steps)
    probe = training_factory()
    try:
        simulator = dict(probe.simulator_metadata)
    finally:
        probe.close()
    metadata: dict[str, object] = {
        "manifest_fingerprint": fingerprint,
        "training_seed": seed,
        "regimen": asdict(regimen),
        "simulator": simulator,
        "configuration": asdict(config),
    }
    metadata = cast(
        dict[str, object], json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
    )
    save_json_artifact(directory / "run.json", metadata)

    def on_update(
        model: PaperRecurrentActorCritic,
        update: RecurrentMAPPOUpdateMetrics,
    ) -> None:
        with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(update), allow_nan=False) + "\n")
        if update.update == 1 or update.update % 10 == 0:
            LOGGER.info(
                "%s/%s seed=%d steps=%d reward=%.5f shape=%.5f",
                experiment.name,
                regimen.name,
                seed,
                update.environment_steps,
                update.mean_rollout_reward,
                update.mean_normalized_shape_rmse,
            )
            save_recurrent_mappo_checkpoint(
                directory / "latest.pt",
                model,
                metadata={**metadata, "environment_steps": update.environment_steps},
            )

    started = perf_counter()
    training = train_recurrent_mappo(
        training_factory,
        config.recurrent,
        seed=seed,
        neighbor_encoder=config.communication.encoder.specification(
            experiment.observation.max_neighbors
        ),
        on_update=on_update,
    )
    training_seconds = perf_counter() - started
    checkpoint = directory / "model.pt"
    save_recurrent_mappo_checkpoint(
        checkpoint,
        training.model,
        metadata={
            **metadata,
            "environment_steps": training.environment_steps,
            "agent_samples": training.agent_samples,
            "observation_profile": config.recurrent.observation_profile.value,
            "action_profile": "direction-speed",
        },
    )
    model, _ = load_recurrent_mappo_checkpoint(checkpoint)
    model.eval()
    condition_episodes: dict[str, list[dict[str, float]]] = {}
    condition_summaries: dict[str, dict[str, float]] = {}
    evaluation_seconds = 0.0
    for condition in _conditions(config, treatment):
        evaluation_regimen = CommunicationRegimen(
            f"evaluate-{condition.name}",
            CommunicationRegimenKind.FIXED,
            (condition,),
        )
        evaluation_factory = _factory(
            config, treatment, physics, evaluation_regimen, training_steps=None
        )
        action_transform: ActionTransform | None = None
        recovery_start: int | None = None
        if treatment is RecurrentTreatment.RECOVERY:
            recovery = config.recovery
            action_transform = velocity_disturbance(
                step=recovery.disturbance_step,
                affected_agents=recovery.affected_agents,
                offset=recovery.velocity_offset,
            )
            recovery_start = recovery.disturbance_step
        started = perf_counter()
        episodes = evaluate_recurrent_environment(
            evaluation_factory,
            RecurrentActorController(model, direction_epsilon=config.recurrent.direction_epsilon),
            episodes=config.communication.mappo.evaluation_episodes,
            seed=config.communication.evaluation_seed,
            horizon=experiment.environment.max_episode_steps,
            time_step_seconds=experiment.environment.time_step_seconds,
            collision_distance_m=experiment.task.collision_distance_m,
            action_transform=action_transform,
            recovery_start_step=recovery_start,
            recovery_threshold=(
                config.recovery.error_threshold if recovery_start is not None else None
            ),
            recovery_hold_steps=config.recovery.threshold_hold_steps,
            episode_metadata=lambda environment: (
                cast(PhysicalEnvironment, environment).episode_metadata
            ),
        )
        evaluation_seconds += perf_counter() - started
        condition_episodes[condition.name] = episodes
        condition_summaries[condition.name] = numeric_mean(episodes)
    return {
        "artifact_schema_version": 1,
        "manifest_fingerprint": fingerprint,
        "training_seed": seed,
        "device": str(training.device),
        "simulator": simulator,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_summary": {
            "environment_steps": float(training.environment_steps),
            "agent_samples": float(training.agent_samples),
            "wall_seconds": training_seconds,
        },
        "condition_episodes": condition_episodes,
        "condition_summaries": condition_summaries,
        "evaluation_wall_seconds": evaluation_seconds,
    }


__all__ = [
    "RecurrentTreatment",
    "recurrent_study_smoke_config",
    "run_recurrent_study",
]
