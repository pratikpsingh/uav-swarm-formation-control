"""Multi-seed controlled evaluation of oracle dynamic-obstacle curricula."""

import hashlib
import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import cast

from uav_swarm_control.algorithms.mappo import (
    load_mappo_checkpoint,
    save_mappo_checkpoint,
    train_mappo,
)
from uav_swarm_control.algorithms.mappo.trainer import MAPPOUpdateMetrics
from uav_swarm_control.configuration.obstacles import (
    CurriculumPhase,
    ObstacleExperimentConfig,
    ObstacleTrainingRegimen,
    TrainingRole,
)
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.obstacles import ObstacleFormationEnvironment
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
from uav_swarm_control.obstacles import ObstacleScenario

LOGGER = logging.getLogger(__name__)
_CONTEXT_PREFIX = "context/"


def obstacle_smoke_config(config: ObstacleExperimentConfig) -> ObstacleExperimentConfig:
    """Bound runtime while retaining three regimens, four scenarios, and five seeds."""
    experiment = replace(
        config.mappo.experiment,
        environment=replace(config.mappo.experiment.environment, max_episode_steps=48),
    )
    algorithm = replace(
        config.mappo.algorithm,
        ppo=replace(
            config.mappo.algorithm.ppo,
            total_steps=256,
            rollout_steps=128,
            minibatch_size=128,
            update_epochs=1,
            device="cpu",
        ),
        num_environments=1,
    )
    return replace(
        config,
        mappo=replace(
            config.mappo,
            experiment=experiment,
            algorithm=algorithm,
            evaluation_episodes=2,
        ),
        physics=PyBulletExperimentConfig(experiment, config.physics.simulator),
        profile="smoke",
    )


def _task_summary(episodes: list[dict[str, float]]) -> dict[str, float]:
    return average_episodes(
        [
            {name: value for name, value in episode.items() if not name.startswith(_CONTEXT_PREFIX)}
            for episode in episodes
        ]
    )


def _difference(right: dict[str, float], left: dict[str, float]) -> dict[str, float]:
    if set(right) != set(left):
        raise ValueError("paired scenario summaries must report identical metrics.")
    return {name: right[name] - left[name] for name in sorted(right)}


def run_obstacle_study(
    config: ObstacleExperimentConfig,
    output: Path,
    *,
    project_root: Path,
    resume: bool = False,
) -> Path:
    """Train all matched regimens and aggregate scenario metrics across seeds."""
    experiment = config.mappo.experiment
    directory = output / config.profile / experiment.name
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "stage10-oracle-dynamic-obstacle-mappo",
        "configuration": asdict(config),
        "provenance": collect_provenance(project_root),
        "design": {
            "controlled_variable": "training obstacle distribution",
            "shared_evaluation_episode_seeds": True,
            "obstacle_model": "kinematic spheres with exact state",
            "contact_model": "sampled sphere overlap at control frequency",
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

    summaries: dict[str, object] = {}
    records_by_role: dict[TrainingRole, list[dict[str, object]]] = {}
    for regimen in config.training_regimens:
        summary, records = _run_regimen(
            config,
            regimen,
            directory / regimen.name,
            manifest["fingerprint"],
            resume=resume,
        )
        summaries[regimen.name] = summary
        records_by_role[regimen.role] = records
    comparisons = _paired_comparisons(config, records_by_role)
    return save_json_artifact(
        directory / "summary.json",
        {
            "artifact_schema_version": 1,
            "manifest_fingerprint": manifest["fingerprint"],
            "profile": config.profile,
            "training_seeds": list(config.training_seeds),
            "evaluation_root_seed": config.evaluation_seed,
            "evaluation_episodes_per_scenario_per_seed": config.mappo.evaluation_episodes,
            "training_regimens": summaries,
            "paired_differences": comparisons,
        },
    )


def _run_regimen(
    config: ObstacleExperimentConfig,
    regimen: ObstacleTrainingRegimen,
    directory: Path,
    fingerprint: object,
    *,
    resume: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    directory.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in config.training_seeds:
        seed_directory = directory / f"seed-{seed}"
        result_path = seed_directory / "result.json"
        checkpoint = seed_directory / "model.pt"
        if seed_directory.exists():
            if not resume or not result_path.is_file() or not checkpoint.is_file():
                raise FileExistsError(f"incomplete seed directory: {seed_directory}")
            saved = cast(dict[str, object], json.loads(result_path.read_text(encoding="utf-8")))
            if (
                saved.get("manifest_fingerprint") != fingerprint
                or saved.get("training_seed") != seed
                or saved.get("checkpoint_sha256")
                != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ):
                raise ValueError(f"completed seed provenance mismatch: {seed_directory}")
            records.append(saved)
            continue
        seed_directory.mkdir(exist_ok=False)
        record = _run_seed(config, regimen, seed, seed_directory, fingerprint)
        save_json_artifact(result_path, record)
        records.append(record)
    scenarios = {
        scenario.value: summarize_seeds(
            [
                cast(dict[str, dict[str, float]], record["scenario_summaries"])[scenario.value]
                for record in records
            ]
        )
        for scenario in config.evaluation_scenarios
    }
    summary: dict[str, object] = {
        "regimen": asdict(regimen),
        "training": summarize_seeds(
            [cast(dict[str, float], record["training_summary"]) for record in records]
        ),
        "scenarios": scenarios,
    }
    save_json_artifact(directory / "summary.json", summary)
    return summary, records


def _run_seed(
    config: ObstacleExperimentConfig,
    regimen: ObstacleTrainingRegimen,
    seed: int,
    directory: Path,
    fingerprint: object,
) -> dict[str, object]:
    experiment = replace(config.mappo.experiment, seed=seed)
    physics = replace(config.physics, experiment=experiment)
    training_steps = math.ceil(
        config.mappo.algorithm.ppo.total_steps / config.mappo.algorithm.num_environments
    )

    def training_factory() -> ObstacleFormationEnvironment:
        return ObstacleFormationEnvironment(
            physics,
            config.pose,
            config.variant,
            config.field,
            regimen.phases,
            training_steps_per_environment=training_steps,
        )

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

    def on_update(model: SharedActorCentralCritic, update: MAPPOUpdateMetrics) -> None:
        with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(update), allow_nan=False) + "\n")
        if update.update == 1 or update.update % 10 == 0:
            LOGGER.info(
                "%s/%s seed=%d steps=%d reward=%.5f",
                experiment.name,
                regimen.name,
                seed,
                update.environment_steps,
                update.mean_rollout_reward,
            )
            save_mappo_checkpoint(
                directory / "latest.pt",
                model,
                metadata={**metadata, "environment_steps": update.environment_steps},
            )

    started = perf_counter()
    training = train_mappo(
        training_factory,
        config.mappo.algorithm,
        seed=seed,
        on_update=on_update,
    )
    training_seconds = perf_counter() - started
    checkpoint = directory / "model.pt"
    save_mappo_checkpoint(
        checkpoint,
        training.model,
        metadata={
            **metadata,
            "environment_steps": training.environment_steps,
            "agent_samples": training.agent_samples,
        },
    )
    model, _ = load_mappo_checkpoint(checkpoint, device="cpu")
    model.eval()

    scenario_episodes: dict[str, list[dict[str, float]]] = {}
    scenario_summaries: dict[str, dict[str, float]] = {}
    evaluation_seconds = 0.0
    for scenario in config.evaluation_scenarios:
        phase = (CurriculumPhase(1.0, scenario),)

        def evaluation_factory(
            active_phases: tuple[CurriculumPhase, ...] = phase,
        ) -> ObstacleFormationEnvironment:
            return ObstacleFormationEnvironment(
                physics,
                config.pose,
                config.variant,
                config.field,
                active_phases,
            )

        started = perf_counter()
        episodes = evaluate_benchmark(
            evaluation_factory,
            ActorController(model),
            episodes=config.mappo.evaluation_episodes,
            seed=config.evaluation_seed,
            horizon=experiment.environment.max_episode_steps,
            time_step_seconds=experiment.environment.time_step_seconds,
            collision_distance_m=experiment.task.collision_distance_m,
            episode_metadata=lambda environment: (
                cast(ObstacleFormationEnvironment, environment).episode_metadata
            ),
            episode_metadata_prefix="context",
        )
        evaluation_seconds += perf_counter() - started
        scenario_episodes[scenario.value] = episodes
        scenario_summaries[scenario.value] = _task_summary(episodes)
    LOGGER.info(
        "%s/%s seed=%d mixed_collision_free_success=%.3f",
        experiment.name,
        regimen.name,
        seed,
        scenario_summaries[ObstacleScenario.MIXED_DYNAMIC.value]["collision_free_success"],
    )
    return {
        "artifact_schema_version": 1,
        "manifest_fingerprint": fingerprint,
        "training_seed": seed,
        "regimen": asdict(regimen),
        "device": str(training.device),
        "simulator": simulator,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_summary": {
            "environment_steps": float(training.environment_steps),
            "agent_samples": float(training.agent_samples),
            "wall_seconds": training_seconds,
        },
        "scenario_episodes": scenario_episodes,
        "scenario_summaries": scenario_summaries,
        "evaluation_wall_seconds": evaluation_seconds,
    }


def _paired_comparisons(
    config: ObstacleExperimentConfig,
    records: dict[TrainingRole, list[dict[str, object]]],
) -> dict[str, object]:
    pairs = (
        (TrainingRole.CURRICULUM, TrainingRole.NO_OBSTACLE_CONTROL),
        (TrainingRole.CURRICULUM, TrainingRole.STATIC_CONTROL),
        (TrainingRole.STATIC_CONTROL, TrainingRole.NO_OBSTACLE_CONTROL),
    )
    result: dict[str, object] = {}
    for right, left in pairs:
        right_records = records[right]
        left_records = records[left]
        if [record["training_seed"] for record in right_records] != [
            record["training_seed"] for record in left_records
        ]:
            raise ValueError("paired comparisons require identical ordered training seeds.")
        result[f"{right.value}-minus-{left.value}"] = {
            scenario.value: summarize_seeds(
                [
                    _difference(
                        cast(dict[str, dict[str, float]], right_record["scenario_summaries"])[
                            scenario.value
                        ],
                        cast(dict[str, dict[str, float]], left_record["scenario_summaries"])[
                            scenario.value
                        ],
                    )
                    for right_record, left_record in zip(right_records, left_records, strict=True)
                ]
            )
            for scenario in config.evaluation_scenarios
        }
    return result


__all__ = ["obstacle_smoke_config", "run_obstacle_study"]
