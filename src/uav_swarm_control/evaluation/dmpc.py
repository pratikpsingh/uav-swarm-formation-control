"""Reproducible DMPC evaluation on the same tasks and metrics used by MAPPO."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import cast

from uav_swarm_control.configuration import DMPCConfig
from uav_swarm_control.configuration.baseline import BaselineConfig
from uav_swarm_control.controllers.dmpc import DistributedMPCController
from uav_swarm_control.environments.formation_progress import FormationProgressEnvironment
from uav_swarm_control.evaluation.artifacts import save_json_artifact
from uav_swarm_control.evaluation.baseline import smoke_config
from uav_swarm_control.evaluation.benchmark import evaluate_benchmark, summarize_records
from uav_swarm_control.evaluation.comparison import comparison_protocol
from uav_swarm_control.evaluation.provenance import (
    collect_provenance,
    stable_digest,
    write_source_snapshot,
)

NATIVE_DMPC_REFERENCE_REVISION = "9ee543d9e2e553708f09f57c7ebc2f55f0302df8"


def dmpc_smoke_config(config: BaselineConfig) -> BaselineConfig:
    """Apply the exact same task/evaluation overrides used by the MAPPO smoke profile."""
    return smoke_config(config)


def run_dmpc(
    task: BaselineConfig,
    controller_config: DMPCConfig,
    output: Path,
    *,
    project_root: Path,
) -> Path:
    """Evaluate DMPC and persist common metrics separately from solver diagnostics."""
    experiment = task.physics.experiment
    if controller_config.limits.minimum_separation_m < experiment.task.collision_distance_m:
        raise ValueError("DMPC minimum separation cannot be below the task collision distance.")
    protocol = comparison_protocol(task)
    directory = output / task.profile / experiment.name / controller_config.name
    if directory.exists():
        raise FileExistsError(f"output exists: {directory}; use a new output root.")
    directory.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, object] = {
        "artifact_schema_version": 1,
        "method": "dmpc-swarm-clean-room-adaptation",
        "exact_native_software_execution": False,
        "native_reference_revision": NATIVE_DMPC_REFERENCE_REVISION,
        "task_configuration": asdict(task),
        "controller_configuration": asdict(controller_config),
        "comparison_protocol": protocol,
        "provenance": collect_provenance(project_root),
    }
    manifest = cast(
        dict[str, object], json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    )
    manifest["fingerprint"] = stable_digest(manifest)
    save_json_artifact(directory / "manifest.json", manifest)
    write_source_snapshot(project_root, directory / "source.zip")

    def factory() -> FormationProgressEnvironment:
        return FormationProgressEnvironment(task.physics)

    probe = factory()
    try:
        simulator = dict(probe.simulator_metadata)
    finally:
        probe.close()
    controller = DistributedMPCController(
        controller_config,
        control_time_step_seconds=experiment.environment.time_step_seconds,
        max_velocity_component_mps=experiment.environment.max_velocity_component_mps,
        max_neighbors=experiment.observation.max_neighbors,
        neighbor_radius_m=experiment.observation.neighbor_radius_m,
    )
    controller_episodes: list[dict[str, float]] = []

    def collect_controller_episode(common: Mapping[str, float]) -> None:
        controller_episodes.append(
            {
                "episode_seed": common["episode_seed"],
                **controller.diagnostics(),
            }
        )

    started = perf_counter()
    episodes = evaluate_benchmark(
        factory,
        controller,
        episodes=task.mappo.evaluation_episodes,
        seed=task.evaluation_seed,
        horizon=experiment.environment.max_episode_steps,
        time_step_seconds=experiment.environment.time_step_seconds,
        collision_distance_m=experiment.task.collision_distance_m,
        on_episode_complete=collect_controller_episode,
    )
    elapsed = perf_counter() - started
    common_records = [
        {key: value for key, value in episode.items() if key != "episode_seed"}
        for episode in episodes
    ]
    diagnostic_records = [
        {key: value for key, value in episode.items() if key != "episode_seed"}
        for episode in controller_episodes
    ]
    return save_json_artifact(
        directory / "result.json",
        {
            "artifact_schema_version": 1,
            "manifest_fingerprint": manifest["fingerprint"],
            "comparison_fingerprint": protocol["fingerprint"],
            "profile": task.profile,
            "method": manifest["method"],
            "exact_native_software_execution": False,
            "information_access": (
                "shared positions, velocities, assigned targets, and deterministic agent IDs"
            ),
            "simulator": simulator,
            "episodes": episodes,
            "common_summary": {
                "uncertainty_unit": "evaluation_episode",
                **summarize_records(common_records, count_key="episode_count"),
            },
            "controller_episodes": controller_episodes,
            "controller_summary": {
                "uncertainty_unit": "evaluation_episode",
                **summarize_records(diagnostic_records, count_key="episode_count"),
            },
            "evaluation_wall_seconds": elapsed,
        },
    )
