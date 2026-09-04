"""Strict YAML loading for experiment configurations."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.models import (
    ConfigurationError,
    EnvironmentConfig,
    ExperimentConfig,
    FormationConfig,
    ObservationConfig,
    Vector3,
)
from uav_swarm_control.formations import FormationKind


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping.")
    source = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for key, item in source.items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{path} keys must be strings.")
        result[key] = item
    return result


def _validate_keys(
    values: Mapping[str, object],
    *,
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = required - values.keys()
    unknown = values.keys() - allowed
    if missing:
        raise ConfigurationError(f"{path} is missing required keys: {', '.join(sorted(missing))}.")
    if unknown:
        raise ConfigurationError(f"{path} contains unknown keys: {', '.join(sorted(unknown))}.")


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer.")
    return value


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number.")
    return float(value)


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{path} must be a string.")
    return value


def _formation_kind(value: object) -> FormationKind:
    text = _string(value, path="formation.kind")
    try:
        return FormationKind(text)
    except ValueError as error:
        choices = ", ".join(kind.value for kind in FormationKind)
        raise ConfigurationError(f"formation.kind must be one of: {choices}.") from error


def _vector3(value: object, *, path: str) -> Vector3:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a YAML list containing three numbers.")
    items = cast(list[object], value)
    if len(items) != 3:
        raise ConfigurationError(f"{path} must be a YAML list containing three numbers.")
    return (
        _number(items[0], path=f"{path}[0]"),
        _number(items[1], path=f"{path}[1]"),
        _number(items[2], path=f"{path}[2]"),
    )


def experiment_config_from_mapping(value: object) -> ExperimentConfig:
    """Build a validated configuration from an untrusted mapping."""
    root = _mapping(value, path="configuration")
    _validate_keys(
        root,
        path="configuration",
        required={
            "schema_version",
            "name",
            "seed",
            "formation",
            "environment",
            "observation",
        },
    )

    formation = _mapping(root["formation"], path="formation")
    _validate_keys(
        formation,
        path="formation",
        required={"kind", "num_agents", "spacing_m"},
        optional={"center_m", "euler_radians"},
    )
    formation_config = FormationConfig(
        kind=_formation_kind(formation["kind"]),
        num_agents=_integer(formation["num_agents"], path="formation.num_agents"),
        spacing_m=_number(formation["spacing_m"], path="formation.spacing_m"),
        center_m=_vector3(formation.get("center_m", [0.0, 0.0, 1.0]), path="formation.center_m"),
        euler_radians=_vector3(
            formation.get("euler_radians", [0.0, 0.0, 0.0]),
            path="formation.euler_radians",
        ),
    )

    environment = _mapping(root["environment"], path="environment")
    _validate_keys(
        environment,
        path="environment",
        required={
            "time_step_seconds",
            "max_episode_steps",
            "max_velocity_component_mps",
        },
    )
    environment_config = EnvironmentConfig(
        time_step_seconds=_number(
            environment["time_step_seconds"],
            path="environment.time_step_seconds",
        ),
        max_episode_steps=_integer(
            environment["max_episode_steps"],
            path="environment.max_episode_steps",
        ),
        max_velocity_component_mps=_number(
            environment["max_velocity_component_mps"],
            path="environment.max_velocity_component_mps",
        ),
    )

    observation = _mapping(root["observation"], path="observation")
    _validate_keys(
        observation,
        path="observation",
        required={"max_neighbors", "neighbor_radius_m"},
    )
    radius_value = observation["neighbor_radius_m"]
    radius = (
        None
        if radius_value is None
        else _number(
            radius_value,
            path="observation.neighbor_radius_m",
        )
    )
    observation_config = ObservationConfig(
        max_neighbors=_integer(observation["max_neighbors"], path="observation.max_neighbors"),
        neighbor_radius_m=radius,
    )

    return ExperimentConfig(
        schema_version=_integer(root["schema_version"], path="schema_version"),
        name=_string(root["name"], path="name"),
        seed=_integer(root["seed"], path="seed"),
        formation=formation_config,
        environment=environment_config,
        observation=observation_config,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Safely load and validate one explicit YAML experiment file."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("experiment configuration must use a .yaml or .yml extension.")
    try:
        contents = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"could not read configuration {config_path}: {error}") from error
    try:
        raw = cast(object, yaml.safe_load(contents))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {config_path}: {error}") from error
    return experiment_config_from_mapping(raw)
