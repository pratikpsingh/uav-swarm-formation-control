"""Validation tests for the controlled Stage 10 protocol."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from uav_swarm_control.configuration import (
    ConfigurationError,
    TrainingRole,
    load_obstacle_experiment_config,
    scenario_for_progress,
)
from uav_swarm_control.obstacles import ObstacleScenario

CONFIG = Path(__file__).parents[2] / "configs/experiment/stage10_dynamic_obstacles_4uav.yaml"


def test_research_protocol_has_matched_controls_and_curriculum() -> None:
    config = load_obstacle_experiment_config(CONFIG)

    assert tuple(regimen.role for regimen in config.training_regimens) == tuple(TrainingRole)
    assert config.evaluation_scenarios == tuple(ObstacleScenario)
    assert len(config.training_seeds) == 5
    assert config.mappo.algorithm.ppo.total_steps == 10_000_000
    curriculum = config.training_regimens[-1]
    assert scenario_for_progress(curriculum.phases, 0.1) is ObstacleScenario.NONE
    assert scenario_for_progress(curriculum.phases, 0.10001) is ObstacleScenario.STATIC
    assert scenario_for_progress(curriculum.phases, 1.0) is ObstacleScenario.MIXED_DYNAMIC


def _raw() -> dict[str, object]:
    return cast(dict[str, object], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def _write(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_missing_control_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    study = cast(dict[str, object], raw["obstacle_study"])
    regimens = cast(list[object], study["training_regimens"])
    regimens.pop()

    with pytest.raises(ConfigurationError, match="exactly one regimen"):
        load_obstacle_experiment_config(_write(tmp_path, raw))


def test_curriculum_order_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    study = cast(dict[str, object], raw["obstacle_study"])
    regimens = cast(list[dict[str, object]], study["training_regimens"])
    phases = cast(list[dict[str, object]], regimens[-1]["phases"])
    phases[1]["scenario"] = "slow-dynamic"

    with pytest.raises(ConfigurationError, match="curriculum requires scenarios"):
        load_obstacle_experiment_config(_write(tmp_path, raw))


def test_neighbor_ablation_is_rejected_in_obstacle_stage(tmp_path: Path) -> None:
    raw = _raw()
    cast(dict[str, object], raw["observation"])["max_neighbors"] = 1

    with pytest.raises(ConfigurationError, match="fixes sensing"):
        load_obstacle_experiment_config(_write(tmp_path, raw))
