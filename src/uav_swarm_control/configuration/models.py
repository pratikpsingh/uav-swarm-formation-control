"""Validated, immutable experiment configuration objects."""

import math
import re
from dataclasses import dataclass, field
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


def _non_negative_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ConfigurationError(f"{name} must be non-negative.")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean.")
    return value


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
class KinematicTaskConfig:
    """Initial-state sampling and episode completion rules."""

    initial_center_m: Vector3 = (-2.0, 0.0, 1.0)
    initial_position_noise_m: float = 0.05
    success_tolerance_m: float = 0.05
    success_hold_steps: int = 5
    collision_distance_m: float = 0.2
    terminate_on_collision: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_center_m",
            _vector3(self.initial_center_m, name="task.initial_center_m"),
        )
        object.__setattr__(
            self,
            "initial_position_noise_m",
            _non_negative_float(
                self.initial_position_noise_m,
                name="task.initial_position_noise_m",
            ),
        )
        object.__setattr__(
            self,
            "success_tolerance_m",
            _finite_float(
                self.success_tolerance_m,
                name="task.success_tolerance_m",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "success_hold_steps",
            _integer(self.success_hold_steps, name="task.success_hold_steps", minimum=1),
        )
        object.__setattr__(
            self,
            "collision_distance_m",
            _finite_float(
                self.collision_distance_m,
                name="task.collision_distance_m",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "terminate_on_collision",
            _boolean(self.terminate_on_collision, name="task.terminate_on_collision"),
        )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Weights for individually reported reward components."""

    navigation_weight: float = 1.0
    formation_weight: float = 0.5
    collision_penalty: float = 5.0
    smoothness_weight: float = 0.05
    success_bonus: float = 10.0

    def __post_init__(self) -> None:
        for field_name in (
            "navigation_weight",
            "formation_weight",
            "collision_penalty",
            "smoothness_weight",
            "success_bonus",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative_float(
                    getattr(self, field_name),
                    name=f"reward.{field_name}",
                ),
            )


@dataclass(frozen=True, slots=True)
class ProportionalControllerConfig:
    """Gain for the deterministic scripted baseline."""

    gain_per_second: float = 1.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gain_per_second",
            _finite_float(
                self.gain_per_second,
                name="controller.gain_per_second",
                positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level configuration required to reproduce one environment run."""

    schema_version: int
    name: str
    seed: int
    formation: FormationConfig
    environment: EnvironmentConfig
    observation: ObservationConfig
    task: KinematicTaskConfig = field(default_factory=KinematicTaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    controller: ProportionalControllerConfig = field(default_factory=ProportionalControllerConfig)

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
        if self.task.collision_distance_m >= self.formation.spacing_m:
            raise ConfigurationError(
                "task.collision_distance_m must be smaller than formation.spacing_m so the "
                "target formation is collision-free."
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
        "task": {
            "initial_center_m": list(config.task.initial_center_m),
            "initial_position_noise_m": config.task.initial_position_noise_m,
            "success_tolerance_m": config.task.success_tolerance_m,
            "success_hold_steps": config.task.success_hold_steps,
            "collision_distance_m": config.task.collision_distance_m,
            "terminate_on_collision": config.task.terminate_on_collision,
        },
        "reward": {
            "navigation_weight": config.reward.navigation_weight,
            "formation_weight": config.reward.formation_weight,
            "collision_penalty": config.reward.collision_penalty,
            "smoothness_weight": config.reward.smoothness_weight,
            "success_bonus": config.reward.success_bonus,
        },
        "controller": {
            "gain_per_second": config.controller.gain_per_second,
        },
    }
