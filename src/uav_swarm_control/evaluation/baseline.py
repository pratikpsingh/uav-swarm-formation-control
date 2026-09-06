"""Auditable multi-seed training and evaluation of the corrected Paper 04 baseline."""

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
from uav_swarm_control.configuration.baseline import BaselineConfig
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.controllers.proportional import ProportionalPositionController
from uav_swarm_control.environments.paper04 import Paper04Environment
from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.benchmark import (
    ActorController,
    average_episodes,
    evaluate_benchmark,
    summarize_seeds,
)
from uav_swarm_control.evaluation.comparison import comparison_protocol
from uav_swarm_control.evaluation.provenance import (
    collect_provenance,
    stable_digest,
    write_source_snapshot,
)
from uav_swarm_control.models import SharedActorCentralCritic

LOGGER = logging.getLogger(__name__)


def smoke_config(config: BaselineConfig) -> BaselineConfig:
    """Shorten budget and horizon explicitly; retain five seeds and network widths."""
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
    mappo = replace(config.mappo, experiment=experiment, algorithm=algorithm, evaluation_episodes=2)
    return replace(
        config,
        mappo=mappo,
        physics=PyBulletExperimentConfig(experiment, config.physics.simulator),
        profile="smoke",
    )


def run_baseline(
    config: BaselineConfig,
    output: Path,
    *,
    project_root: Path,
    resume: bool = False,
) -> Path:
    """Train seeds and persist episode records before aggregating seed-level means.

    Resume skips completed runs with matching source/configuration/checkpoint hashes.
    Interrupted training must restart in a new output: optimizer state is not resumable.
    Latest weights remain available for diagnosis.
    """
    experiment = config.mappo.experiment
    directory = output / config.profile / experiment.name
    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "corrected-paper04-feedforward-mappo",
        "exact_paper_reproduction": False,
        "configuration": asdict(config),
        "provenance": collect_provenance(project_root),
        "comparison_protocol": comparison_protocol(config),
    }
    # Canonical JSON converts dataclass tuples to lists before resume comparison.
    manifest = cast(dict[str, object], json.loads(json.dumps(manifest, allow_nan=False)))
    manifest["fingerprint"] = stable_digest(manifest)
    manifest_path = directory / "manifest.json"
    if directory.exists():
        if not resume:
            raise FileExistsError(f"output exists: {directory}; use --resume or a new output.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "resume requires identical configuration, source, and runtime provenance."
            )
    else:
        directory.mkdir(parents=True, exist_ok=False)
        save_json_artifact(manifest_path, manifest)
        write_source_snapshot(project_root, directory / "source.zip")

    records: list[dict[str, float]] = []
    for seed in config.training_seeds:
        seed_directory = directory / f"seed-{seed}"
        result_path = seed_directory / "result.json"
        checkpoint = seed_directory / "model.pt"
        if seed_directory.exists():
            if not resume or not result_path.is_file():
                raise FileExistsError(
                    f"incomplete seed directory: {seed_directory}; use a new output."
                )
            saved = cast(dict[str, object], json.loads(result_path.read_text(encoding="utf-8")))
            if (
                saved.get("fingerprint") != manifest["fingerprint"]
                or saved.get("training_seed") != seed
                or saved.get("checkpoint_sha256")
                != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ):
                raise ValueError(f"completed seed provenance mismatch: {seed_directory}")
            records.append(cast(dict[str, float], saved["summary"]))
            continue
        seed_directory.mkdir(exist_ok=False)
        record = _run_seed(config, seed, seed_directory, manifest["fingerprint"])
        save_json_artifact(result_path, record)
        records.append(cast(dict[str, float], record["summary"]))
    return save_json_artifact(
        directory / "summary.json",
        {
            "fingerprint": manifest["fingerprint"],
            "comparison_fingerprint": cast(dict[str, object], manifest["comparison_protocol"])[
                "fingerprint"
            ],
            "profile": config.profile,
            "exact_paper_reproduction": False,
            "training_seeds": list(config.training_seeds),
            "evaluation_root_seed": config.evaluation_seed,
            "evaluation_episodes_per_seed": config.mappo.evaluation_episodes,
            **summarize_seeds(records),
        },
    )


def _run_seed(
    config: BaselineConfig, seed: int, directory: Path, fingerprint: object
) -> dict[str, object]:
    experiment = replace(config.mappo.experiment, seed=seed)
    physics = replace(config.physics, experiment=experiment)

    def factory() -> Paper04Environment:
        return Paper04Environment(physics)

    probe = factory()
    try:
        simulator = dict(probe.simulator_metadata)
    finally:
        probe.close()
    metadata: dict[str, object] = {
        "manifest_fingerprint": fingerprint,
        "training_seed": seed,
        "simulator": simulator,
        "configuration": asdict(
            replace(config, physics=physics, mappo=replace(config.mappo, experiment=experiment))
        ),
    }
    metadata = cast(dict[str, object], json.loads(json.dumps(metadata, allow_nan=False)))
    save_json_artifact(directory / "run.json", metadata)

    def on_update(model: SharedActorCentralCritic, update: MAPPOUpdateMetrics) -> None:
        with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(update), allow_nan=False) + "\n")
        if update.update == 1 or update.update % 10 == 0:
            LOGGER.info(
                "%s seed=%d steps=%d reward=%.5f",
                experiment.name,
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
    training = train_mappo(factory, config.mappo.algorithm, seed=seed, on_update=on_update)
    train_seconds = perf_counter() - started
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

    def evaluate(
        controller: ActorController | ProportionalPositionController,
    ) -> list[dict[str, float]]:
        return evaluate_benchmark(
            factory,
            controller,
            episodes=config.mappo.evaluation_episodes,
            seed=config.evaluation_seed,
            horizon=experiment.environment.max_episode_steps,
            time_step_seconds=experiment.environment.time_step_seconds,
            collision_distance_m=experiment.task.collision_distance_m,
        )

    started = perf_counter()
    episodes = evaluate(ActorController(model))
    evaluation_seconds = perf_counter() - started
    scripted = evaluate(
        ProportionalPositionController(
            experiment.controller.gain_per_second,
            experiment.environment.max_velocity_component_mps,
        )
    )
    summary = average_episodes(episodes)
    summary.update(
        {
            "training_environment_steps": float(training.environment_steps),
            "training_agent_samples": float(training.agent_samples),
            "training_wall_seconds": train_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
        }
    )
    LOGGER.info("%s seed=%d complete success=%.3f", experiment.name, seed, summary["success"])
    return {
        "fingerprint": fingerprint,
        "training_seed": seed,
        "device": str(training.device),
        "simulator": simulator,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "episodes": episodes,
        "scripted_episodes": scripted,
        "scripted_summary": average_episodes(scripted),
        "summary": summary,
    }


def evaluate_saved_baseline(
    config: BaselineConfig, checkpoint: Path, output: Path, *, project_root: Path
) -> Path:
    """Re-evaluate a saved actor with an explicitly compatible task and simulator."""
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    model, metadata = load_mappo_checkpoint(checkpoint, device="cpu")
    saved_configuration = metadata.get("configuration")
    if not isinstance(saved_configuration, dict):
        raise ValueError("checkpoint does not contain baseline configuration metadata.")
    saved_configuration = cast(dict[str, object], saved_configuration)
    training_seed = metadata.get("training_seed")
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        raise ValueError("checkpoint training seed is missing or invalid.")
    physics = replace(
        config.physics, experiment=replace(config.physics.experiment, seed=training_seed)
    )
    expected = json.loads(json.dumps(asdict(physics)))
    if saved_configuration.get("physics") != expected:
        raise ValueError("checkpoint task/simulator differs from evaluation configuration.")
    model.eval()

    def factory() -> Paper04Environment:
        return Paper04Environment(physics)

    probe = factory()
    try:
        simulator = dict(probe.simulator_metadata)
    finally:
        probe.close()
    if metadata.get("simulator") != simulator:
        raise ValueError("checkpoint simulator provenance differs from installed simulator.")
    started = perf_counter()
    episodes = evaluate_benchmark(
        factory,
        ActorController(model),
        episodes=config.mappo.evaluation_episodes,
        seed=config.evaluation_seed,
        horizon=physics.experiment.environment.max_episode_steps,
        time_step_seconds=physics.experiment.environment.time_step_seconds,
        collision_distance_m=physics.experiment.task.collision_distance_m,
    )
    return save_json_artifact(
        output,
        {
            "artifact_schema_version": 1,
            "profile": config.profile,
            "exact_paper_reproduction": False,
            "training_seed": training_seed,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "configuration": asdict(config),
            "simulator": simulator,
            "evaluation_provenance": collect_provenance(project_root),
            "episodes": episodes,
            "summary": average_episodes(episodes),
            "evaluation_wall_seconds": perf_counter() - started,
        },
    )
