"""Reproducible multi-seed training and held-out evaluation for Stage 9."""

import hashlib
import json
import logging
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
from uav_swarm_control.configuration.generalization import (
    GeneralizationConfig,
    GeneralizationVariant,
)
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.generalization import GeneralizedFormationEnvironment
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
from uav_swarm_control.formations import FormationPoseRange
from uav_swarm_control.models import SharedActorCentralCritic
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed

LOGGER = logging.getLogger(__name__)
_POSE_PREFIX = "pose/"


def generalization_smoke_config(config: GeneralizationConfig) -> GeneralizationConfig:
    """Bound runtime while retaining all variants, five seeds and both pose splits."""
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


def _common_summary(episodes: list[dict[str, float]]) -> dict[str, float]:
    records = [
        {name: value for name, value in episode.items() if not name.startswith(_POSE_PREFIX)}
        for episode in episodes
    ]
    return average_episodes(records)


def _generalization_gap(
    in_distribution: dict[str, float], held_out: dict[str, float]
) -> dict[str, float]:
    if set(in_distribution) != set(held_out):
        raise ValueError("evaluation splits must report identical common metrics.")
    return {name: held_out[name] - in_distribution[name] for name in sorted(in_distribution)}


def run_generalization(
    config: GeneralizationConfig,
    output: Path,
    *,
    project_root: Path,
    resume: bool = False,
) -> Path:
    """Train every configured variant and aggregate in/held-out metrics across seeds."""
    experiment = config.mappo.experiment
    directory = output / config.profile / experiment.name
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "stage9-3d-generalized-feedforward-mappo",
        "configuration": asdict(config),
        "provenance": collect_provenance(project_root),
        "split_semantics": {
            "in_distribution": "fresh seeds sampled from training_pose",
            "held_out": "fresh seeds sampled from disjoint held_out_pose",
        },
    }
    manifest = cast(
        dict[str, object],
        json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False)),
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

    variant_summaries: dict[str, object] = {}
    for variant in config.variants:
        variant_summaries[variant.name] = _run_variant(
            config,
            variant,
            directory / variant.name,
            manifest["fingerprint"],
            resume=resume,
        )
    return save_json_artifact(
        directory / "summary.json",
        {
            "artifact_schema_version": 1,
            "manifest_fingerprint": manifest["fingerprint"],
            "profile": config.profile,
            "training_seeds": list(config.training_seeds),
            "evaluation_root_seed": config.evaluation_seed,
            "evaluation_episodes_per_split_per_seed": config.mappo.evaluation_episodes,
            "variants": variant_summaries,
        },
    )


def _run_variant(
    config: GeneralizationConfig,
    variant: GeneralizationVariant,
    directory: Path,
    fingerprint: object,
    *,
    resume: bool,
) -> dict[str, object]:
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
        record = _run_seed(config, variant, seed, seed_directory, fingerprint)
        save_json_artifact(result_path, record)
        records.append(record)

    training = [cast(dict[str, float], record["training_summary"]) for record in records]
    split_summaries = {
        split: summarize_seeds(
            [
                cast(dict[str, dict[str, float]], record["split_summaries"])[split]
                for record in records
            ]
        )
        for split in ("in_distribution", "held_out")
    }
    gaps = summarize_seeds(
        [cast(dict[str, float], record["generalization_gap"]) for record in records]
    )
    summary: dict[str, object] = {
        "variant": asdict(variant),
        "training": summarize_seeds(training),
        "splits": split_summaries,
        "generalization_gap_held_out_minus_in_distribution": gaps,
    }
    save_json_artifact(directory / "summary.json", summary)
    return summary


def _run_seed(
    config: GeneralizationConfig,
    variant: GeneralizationVariant,
    seed: int,
    directory: Path,
    fingerprint: object,
) -> dict[str, object]:
    experiment = replace(config.mappo.experiment, seed=seed)
    physics = replace(config.physics, experiment=experiment)

    def factory(pose_range: FormationPoseRange) -> GeneralizedFormationEnvironment:
        return GeneralizedFormationEnvironment(physics, pose_range, variant)

    probe = factory(config.training_pose)
    try:
        simulator = dict(probe.simulator_metadata)
    finally:
        probe.close()
    metadata: dict[str, object] = {
        "manifest_fingerprint": fingerprint,
        "training_seed": seed,
        "variant": asdict(variant),
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
                variant.name,
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
        lambda: factory(config.training_pose),
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

    split_episodes: dict[str, list[dict[str, float]]] = {}
    split_summaries: dict[str, dict[str, float]] = {}
    evaluation_seconds = 0.0
    for split_index, (split, pose_range) in enumerate(
        (("in_distribution", config.training_pose), ("held_out", config.held_out_pose))
    ):

        def split_factory(
            active_pose_range: FormationPoseRange = pose_range,
        ) -> GeneralizedFormationEnvironment:
            return factory(active_pose_range)

        started = perf_counter()
        episodes = evaluate_benchmark(
            split_factory,
            ActorController(model),
            episodes=config.mappo.evaluation_episodes,
            seed=derive_indexed_seed(
                config.evaluation_seed,
                RandomStream.EVALUATION,
                split_index,
            ),
            horizon=experiment.environment.max_episode_steps,
            time_step_seconds=experiment.environment.time_step_seconds,
            collision_distance_m=experiment.task.collision_distance_m,
            episode_metadata=lambda environment: (
                cast(GeneralizedFormationEnvironment, environment).episode_metadata
            ),
        )
        evaluation_seconds += perf_counter() - started
        split_episodes[split] = episodes
        split_summaries[split] = _common_summary(episodes)

    LOGGER.info(
        "%s/%s seed=%d held_out_success=%.3f",
        experiment.name,
        variant.name,
        seed,
        split_summaries["held_out"]["success"],
    )
    return {
        "artifact_schema_version": 1,
        "manifest_fingerprint": fingerprint,
        "training_seed": seed,
        "variant": asdict(variant),
        "device": str(training.device),
        "simulator": simulator,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_summary": {
            "environment_steps": float(training.environment_steps),
            "agent_samples": float(training.agent_samples),
            "wall_seconds": training_seconds,
        },
        "split_episodes": split_episodes,
        "split_summaries": split_summaries,
        "generalization_gap": _generalization_gap(
            split_summaries["in_distribution"], split_summaries["held_out"]
        ),
        "evaluation_wall_seconds": evaluation_seconds,
    }


__all__ = ["generalization_smoke_config", "run_generalization"]
