"""Reproducible, machine-readable evaluation artifacts."""

import json
from collections.abc import Mapping
from pathlib import Path

from uav_swarm_control.configuration import (
    PyBulletExperimentConfig,
    pybullet_experiment_config_to_dict,
)
from uav_swarm_control.environments.drone_backend import SimulatorMetadataValue
from uav_swarm_control.evaluation.rollout import EpisodeResult


def pybullet_episode_record(
    config: PyBulletExperimentConfig,
    simulator_metadata: Mapping[str, SimulatorMetadataValue],
    result: EpisodeResult,
) -> dict[str, object]:
    """Build a complete record for one scripted simulator episode."""
    return {
        "artifact_schema_version": 1,
        "experiment": pybullet_experiment_config_to_dict(config),
        "simulator": dict(simulator_metadata),
        "result": {
            "steps": result.steps,
            "returns": result.returns.tolist(),
            "terminated": result.terminated,
            "truncated": result.truncated,
            "success": result.success,
            "final_metrics": dict(result.final_metrics),
        },
    }


def save_json_artifact(path: str | Path, record: Mapping[str, object]) -> Path:
    """Atomically save a stable, human-readable JSON artifact."""
    output_path = Path(path)
    if output_path.suffix != ".json":
        raise ValueError("evaluation artifact path must end in .json.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(dict(record), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path
