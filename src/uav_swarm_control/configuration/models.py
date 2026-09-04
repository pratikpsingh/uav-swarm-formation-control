"""Validated, immutable experiment configuration objects."""

import math
import re
from dataclasses import dataclass
from typing import cast

from uav_swarm_control.formations import FormationKind, create_formation
from uav_swarm_control.seeding import validate_seed

type Vector3 = tuple[float, float, float]

_EXPERIMENT_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates its schema."""


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer.")
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}.")
    return value


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return result


def _vector3(value: object, *, name: str) -> Vector3:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{name} must contain exactly three values.")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != 3:
        raise ConfigurationError(f"{name} must contain exactly three values.")
    return (
        _finite_float(items[0], name=f"{name}[0]"),
        _finite_float(items[1], name=f"{name}[1]"),
        _finite_float(items[2], name=f"{name}[2]"),
    )


def _formation_kind(value: object) -> FormationKind:
    try:
        return FormationKind(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(kind.value for kind in FormationKind)
        raise ConfigurationError(f"formation.kind must be one of: {choices}.") from error


@dataclass(frozen=True, slots=True)
class FormationConfig:
    """Target formation geometry and world pose."""

    kind: FormationKind
    num_agents: int
    spacing_m: float
    center_m: Vector3 = (0.0, 0.0, 1.0)
    euler_radians: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        kind = _formation_kind(self.kind)
        count = _integer(self.num_agents, name="formation.num_agents", minimum=1)
        spacing = _finite_float(self.spacing_m, name="formation.spacing_m", positive=True)
        center = _vector3(self.center_m, name="formation.center_m")
        euler = _vector3(self.euler_radians, name="formation.euler_radians")
        try:
            create_formation(kind, count, spacing)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "num_agents", count)
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "euler_radians", euler)


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Timing and physical limits shared by environment implementations."""

    time_step_seconds: float
    max_episode_steps: int
    max_velocity_component_mps: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_step_seconds",
            _finite_float(
                self.time_step_seconds,
                name="environment.time_step_seconds",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_episode_steps",
            _integer(
                self.max_episode_steps,
                name="environment.max_episode_steps",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "max_velocity_component_mps",
            _finite_float(
                self.max_velocity_component_mps,
                name="environment.max_velocity_component_mps",
                positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Capacity and sensing limits for decentralized local observations."""

    max_neighbors: int
    neighbor_radius_m: float | None

    def __post_init__(self) -> None:
        max_neighbors = _integer(
            self.max_neighbors,
            name="observation.max_neighbors",
            minimum=0,
        )
        radius = self.neighbor_radius_m
        if radius is not None:
            radius = _finite_float(
                radius,
                name="observation.neighbor_radius_m",
                positive=True,
            )
        object.__setattr__(self, "max_neighbors", max_neighbors)
        object.__setattr__(self, "neighbor_radius_m", radius)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level configuration required to reproduce one environment run."""

    schema_version: int
    name: str
    seed: int
    formation: FormationConfig
    environment: EnvironmentConfig
    observation: ObservationConfig

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, name="schema_version", minimum=1)
        if version != 1:
            raise ConfigurationError(f"unsupported schema_version {version}; expected 1.")
        if not _EXPERIMENT_NAME.fullmatch(self.name):
            raise ConfigurationError(
                "name must be a lowercase kebab-case identifier, for example "
                "'triangle-kinematic-smoke'."
            )
        try:
            seed = validate_seed(self.seed)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error
        available_neighbors = self.formation.num_agents - 1
        if self.observation.max_neighbors > available_neighbors:
            raise ConfigurationError(
                "observation.max_neighbors cannot exceed num_agents - 1; "
                f"received {self.observation.max_neighbors} for "
                f"{self.formation.num_agents} agents."
            )
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "seed", seed)


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    """Convert a validated configuration into a serializable resolved mapping."""
    return {
        "schema_version": config.schema_version,
        "name": config.name,
        "seed": config.seed,
        "formation": {
            "kind": config.formation.kind.value,
            "num_agents": config.formation.num_agents,
            "spacing_m": config.formation.spacing_m,
            "center_m": list(config.formation.center_m),
            "euler_radians": list(config.formation.euler_radians),
        },
        "environment": {
            "time_step_seconds": config.environment.time_step_seconds,
            "max_episode_steps": config.environment.max_episode_steps,
            "max_velocity_component_mps": config.environment.max_velocity_component_mps,
        },
        "observation": {
            "max_neighbors": config.observation.max_neighbors,
            "neighbor_radius_m": config.observation.neighbor_radius_m,
        },
    }
