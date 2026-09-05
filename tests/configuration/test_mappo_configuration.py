"""Tests for strict MAPPO and kinematic-task composition."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from uav_swarm_control.configuration import (
    ConfigurationError,
    load_mappo_experiment_config,
    mappo_experiment_config_from_mapping,
)


def test_sample_mappo_configuration_loads_and_counts_samples() -> None:
    path = Path(__file__).parents[2] / "configs" / "experiment" / "mappo_triangle_kinematic.yaml"

    config = load_mappo_experiment_config(path)

    assert config.experiment.formation.num_agents == 3
    assert config.algorithm.num_environments == 4
    assert config.algorithm.num_updates == 64
    assert config.algorithm.ppo.hidden_sizes == (64, 64)
    assert config.algorithm.critic_hidden_sizes == (128, 128)


def test_mappo_configuration_requires_algorithm_and_evaluation() -> None:
    with pytest.raises(ConfigurationError, match="requires algorithm and evaluation"):
        mappo_experiment_config_from_mapping({"schema_version": 1})


def test_mappo_configuration_rejects_unknown_algorithm_key() -> None:
    path = Path(__file__).parents[2] / "configs" / "experiment" / "mappo_triangle_kinematic.yaml"
    values = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    assert isinstance(values, dict)
    root = cast(dict[str, object], values)
    algorithm = root["algorithm"]
    assert isinstance(algorithm, dict)
    cast(dict[str, object], algorithm)["unknown"] = 1

    with pytest.raises(ConfigurationError, match="unknown keys: unknown"):
        mappo_experiment_config_from_mapping(root)
