"""Multi-seed fixed-versus-variable neighbor-topology study."""

import hashlib
import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
from typing import cast

from uav_swarm_control.algorithms.mappo import (
    load_mappo_checkpoint,
    save_mappo_checkpoint,
    train_mappo,
)
from uav_swarm_control.algorithms.mappo.trainer import MAPPOUpdateMetrics
from uav_swarm_control.communication import (
    CommunicationRegimen,
    CommunicationRegimenKind,
)
from uav_swarm_control.configuration.communication import CommunicationExperimentConfig
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
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

LOGGER = logging.getLogger(__name__)
_CONTEXT_PREFIX = "context/"


def communication_smoke_config(
    config: CommunicationExperimentConfig,
) -> CommunicationExperimentConfig:
    """Bound runtime while retaining every factor, regimen, and training seed."""
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
        raise ValueError("paired condition summaries must report identical metrics.")
    return {name: right[name] - left[name] for name in sorted(right)}


def student_t_critical_95(sample_count: int) -> float:
    """Approximate the two-sided 95% Student-t critical value."""
    degrees = sample_count - 1
    if degrees < 1:
        raise ValueError("confidence intervals require at least two samples.")
    if degrees == 4:
        return 2.7764451051977987
    z = NormalDist().inv_cdf(0.975)
    inverse = 1.0 / degrees
    return (
        z
        + (z**3 + z) * inverse / 4.0
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) * inverse**2 / 96.0
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) * inverse**3 / 384.0
    )


def _summarize_policy_seeds(
    records: list[dict[str, float]],
) -> dict[str, object]:
    """Add a two-sided 95% t interval to the existing across-seed summary."""
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
        "method": "two-sided Student-t interval across independent training seeds",
        "critical_value": critical,
    }
    return summary


def run_communication_study(
    config: CommunicationExperimentConfig,
    output: Path,
    *,
    project_root: Path,
    resume: bool = False,
) -> Path:
    """Train fixed and variable policies and evaluate all topology conditions."""
    experiment = config.mappo.experiment
    directory = output / config.profile / experiment.name
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "neighbor-encoder-mappo",
        "configuration": asdict(config),
        "provenance": collect_provenance(project_root),
        "design": {
            "controlled_variables": [
                "requested neighbor count",
                "sensing radius",
                "obstacle condition",
                "formation and swarm size across configuration files",
            ],
            "shared_evaluation_episode_seeds": True,
            "actor_neighbor_aggregation": "shared encoder plus masked mean",
            "communication_byte_model": (
                "received neighbor payload only; protocol headers, retransmission, "
                "localization, and obstacle sensing are excluded"
            ),
            "paper02_alignment": {
                "paper02_factors": {
                    "robots": 32,
                    "sensed_neighbors": [1, 2, 6, 16, 31],
                },
                "comparable_concepts": [
                    "success",
                    "collision occurrence",
                    "terminal goal error",
                    "neighbor-count trend",
                ],
                "not_directly_comparable": [
                    "reward totals",
                    "absolute metric values across different tasks",
                    "rotor-thrust policy versus velocity-command policy",
                    "Paper 02 static clutter versus this study's dynamic obstacles",
                    "attention encoder versus masked-mean encoder",
                ],
            },
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
    records_by_name: dict[str, list[dict[str, object]]] = {}
    for regimen in config.training_regimens:
        summary, records = _run_regimen(
            config,
            regimen,
            directory / regimen.name,
            manifest["fingerprint"],
            resume=resume,
        )
        summaries[regimen.name] = summary
        records_by_name[regimen.name] = records
    return save_json_artifact(
        directory / "summary.json",
        {
            "artifact_schema_version": 1,
            "manifest_fingerprint": manifest["fingerprint"],
            "profile": config.profile,
            "training_seeds": list(config.training_seeds),
            "evaluation_root_seed": config.evaluation_seed,
            "evaluation_episodes_per_condition_per_seed": config.mappo.evaluation_episodes,
            "training_regimens": summaries,
            "paired_differences": _paired_comparisons(config, records_by_name),
            "interpretation_rule": (
                "Infer effects from physical metrics and graph descriptors with seed "
                "variation; never compare reward totals with Paper 02."
            ),
        },
    )


def _run_regimen(
    config: CommunicationExperimentConfig,
    regimen: CommunicationRegimen,
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
    condition_summaries = {
        condition.name: _summarize_policy_seeds(
            [
                cast(dict[str, dict[str, float]], record["condition_summaries"])[condition.name]
                for record in records
            ]
        )
        for condition in config.evaluation_conditions
    }
    summary: dict[str, object] = {
        "regimen": asdict(regimen),
        "training": summarize_seeds(
            [cast(dict[str, float], record["training_summary"]) for record in records]
        ),
        "conditions": condition_summaries,
    }
    save_json_artifact(directory / "summary.json", summary)
    return summary, records


def _run_seed(
    config: CommunicationExperimentConfig,
    regimen: CommunicationRegimen,
    seed: int,
    directory: Path,
    fingerprint: object,
) -> dict[str, object]:
    experiment = replace(config.mappo.experiment, seed=seed)
    physics = replace(config.physics, experiment=experiment)
    training_steps = math.ceil(
        config.mappo.algorithm.ppo.total_steps / config.mappo.algorithm.num_environments
    )

    def training_factory() -> CommunicationFormationEnvironment:
        return CommunicationFormationEnvironment(
            physics,
            config.pose,
            config.variant,
            config.field,
            regimen,
            payload_bytes_per_neighbor=config.encoder.payload_bytes_per_neighbor,
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
        neighbor_encoder=config.encoder.specification(experiment.observation.max_neighbors),
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

    condition_episodes: dict[str, list[dict[str, float]]] = {}
    condition_summaries: dict[str, dict[str, float]] = {}
    evaluation_seconds = 0.0
    for condition in config.evaluation_conditions:
        evaluation_regimen = CommunicationRegimen(
            f"evaluate-{condition.name}",
            CommunicationRegimenKind.FIXED,
            (condition,),
        )

        def evaluation_factory(
            active_regimen: CommunicationRegimen = evaluation_regimen,
        ) -> CommunicationFormationEnvironment:
            return CommunicationFormationEnvironment(
                physics,
                config.pose,
                config.variant,
                config.field,
                active_regimen,
                payload_bytes_per_neighbor=config.encoder.payload_bytes_per_neighbor,
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
                cast(CommunicationFormationEnvironment, environment).episode_metadata
            ),
            episode_metadata_prefix="context",
        )
        evaluation_seconds += perf_counter() - started
        condition_episodes[condition.name] = episodes
        condition_summaries[condition.name] = _task_summary(episodes)
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
        "condition_episodes": condition_episodes,
        "condition_summaries": condition_summaries,
        "evaluation_wall_seconds": evaluation_seconds,
    }


def _paired_comparisons(
    config: CommunicationExperimentConfig,
    records: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    variable = next(
        regimen
        for regimen in config.training_regimens
        if regimen.kind is CommunicationRegimenKind.VARIABLE
    )
    result: dict[str, object] = {}
    for fixed in (
        regimen
        for regimen in config.training_regimens
        if regimen.kind is CommunicationRegimenKind.FIXED
    ):
        right_records = records[variable.name]
        left_records = records[fixed.name]
        if [record["training_seed"] for record in right_records] != [
            record["training_seed"] for record in left_records
        ]:
            raise ValueError("paired comparisons require identical ordered training seeds.")
        result[f"{variable.name}-minus-{fixed.name}"] = {
            condition.name: _summarize_policy_seeds(
                [
                    _difference(
                        cast(dict[str, dict[str, float]], right["condition_summaries"])[
                            condition.name
                        ],
                        cast(dict[str, dict[str, float]], left["condition_summaries"])[
                            condition.name
                        ],
                    )
                    for right, left in zip(right_records, left_records, strict=True)
                ]
            )
            for condition in config.evaluation_conditions
        }
    return result


__all__ = [
    "communication_smoke_config",
    "run_communication_study",
    "student_t_critical_95",
]
