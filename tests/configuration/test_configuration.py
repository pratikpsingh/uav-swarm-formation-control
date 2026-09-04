"""Tests for strict experiment configuration."""

from pathlib import Path

import pytest

from uav_swarm_control.configuration import (
    ConfigurationError,
    experiment_config_from_mapping,
    experiment_config_to_dict,
    load_experiment_config,
)
from uav_swarm_control.formations import FormationKind


def _valid_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "triangle-kinematic-test",
        "seed": 42,
        "formation": {
            "kind": "triangle",
            "num_agents": 3,
            "spacing_m": 1.0,
            "center_m": [0.0, 0.0, 1.0],
            "euler_radians": [0.0, 0.0, 0.0],
        },
        "environment": {
            "time_step_seconds": 0.1,
            "max_episode_steps": 100,
            "max_velocity_component_mps": 2.0,
        },
        "observation": {"max_neighbors": 2, "neighbor_radius_m": None},
    }


def test_mapping_builds_typed_immutable_configuration() -> None:
    config = experiment_config_from_mapping(_valid_mapping())

    assert config.formation.kind is FormationKind.TRIANGLE
    assert config.formation.center_m == (0.0, 0.0, 1.0)
    assert config.environment.max_episode_steps == 100
    assert config.observation.neighbor_radius_m is None


def test_resolved_mapping_round_trip_preserves_configuration() -> None:
    config = experiment_config_from_mapping(_valid_mapping())

    assert experiment_config_from_mapping(experiment_config_to_dict(config)) == config


def test_repository_example_configuration_loads() -> None:
    path = Path(__file__).parents[2] / "configs" / "experiment" / "triangle_kinematic.yaml"
    config = load_experiment_config(path)

    assert config.name == "triangle-kinematic-smoke"
    assert config.seed == 20260905
    assert config.task.success_hold_steps == 5
    assert config.reward.success_bonus == 10.0
    assert config.controller.gain_per_second == 1.5


def test_optional_formation_pose_has_explicit_defaults() -> None:
    values = _valid_mapping()
    formation = values["formation"]
    assert isinstance(formation, dict)
    del formation["center_m"]
    del formation["euler_radians"]

    config = experiment_config_from_mapping(values)

    assert config.formation.center_m == (0.0, 0.0, 1.0)
    assert config.formation.euler_radians == (0.0, 0.0, 0.0)


def test_unknown_keys_are_rejected() -> None:
    values = _valid_mapping()
    values["typo"] = 123

    with pytest.raises(ConfigurationError, match="unknown keys: typo"):
        experiment_config_from_mapping(values)


def test_neighbor_capacity_cannot_exceed_available_agents() -> None:
    values = _valid_mapping()
    observation = values["observation"]
    assert isinstance(observation, dict)
    observation["max_neighbors"] = 3

    with pytest.raises(ConfigurationError, match="num_agents - 1"):
        experiment_config_from_mapping(values)


def test_collision_distance_must_keep_target_formation_collision_free() -> None:
    values = _valid_mapping()
    values["task"] = {"collision_distance_m": 1.0}

    with pytest.raises(ConfigurationError, match="target formation is collision-free"):
        experiment_config_from_mapping(values)


def test_formation_count_must_match_selected_geometry() -> None:
    values = _valid_mapping()
    formation = values["formation"]
    assert isinstance(formation, dict)
    formation["kind"] = "square"

    with pytest.raises(ConfigurationError, match="num_agents must be 4"):
        experiment_config_from_mapping(values)


def test_unknown_formation_kind_uses_configuration_error() -> None:
    values = _valid_mapping()
    formation = values["formation"]
    assert isinstance(formation, dict)
    formation["kind"] = "hexagon"

    with pytest.raises(ConfigurationError, match=r"formation.kind must be one of"):
        experiment_config_from_mapping(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_step_seconds", 0.0),
        ("max_episode_steps", 0),
        ("max_velocity_component_mps", -1.0),
    ],
)
def test_invalid_environment_limits_are_rejected(field: str, value: float) -> None:
    values = _valid_mapping()
    environment = values["environment"]
    assert isinstance(environment, dict)
    environment[field] = value

    with pytest.raises(ConfigurationError):
        experiment_config_from_mapping(values)
