"""Tests for complete and durable simulator result records."""

import json
from pathlib import Path

import numpy as np

from uav_swarm_control.configuration import load_pybullet_experiment_config
from uav_swarm_control.evaluation.artifacts import pybullet_episode_record, save_json_artifact
from uav_swarm_control.evaluation.rollout import EpisodeResult


def test_saves_config_provenance_and_metrics_atomically(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[2] / "configs" / "experiment" / "pybullet_hover.yaml"
    config = load_pybullet_experiment_config(config_path)
    result = EpisodeResult(
        steps=24,
        returns=np.array([1.5], dtype=np.float32),
        terminated=True,
        truncated=False,
        success=True,
        final_metrics={"position_rmse_m": 0.01},
    )
    record = pybullet_episode_record(
        config,
        {"simulator_revision": "abc", "physics_frequency_hz": 240},
        result,
    )

    output = save_json_artifact(tmp_path / "nested" / "result.json", record)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["experiment"]["simulator"]["physics_steps_per_control"] == 5
    assert saved["simulator"]["simulator_revision"] == "abc"
    assert saved["result"]["final_metrics"]["position_rmse_m"] == 0.01
    assert not output.with_suffix(".json.tmp").exists()
