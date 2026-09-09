"""Strict configuration for recurrent physical research treatments."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.communication import (
    CommunicationExperimentConfig,
    communication_experiment_config_from_mapping,
)
from uav_swarm_control.configuration.models import ConfigurationError
from uav_swarm_control.configuration.recurrent_mappo import (
    RecurrentMAPPOConfig,
    recurrent_settings_from_mapping,
)
from uav_swarm_control.formations import FormationKind
from uav_swarm_control.formations.equal_size import equal_size_formation


@dataclass(frozen=True, slots=True)
class MissionStudyConfig:
    initial_ground_altitude_m: float
    construction_altitude_m: float
    construction_hold_steps: int
    waypoint_spacing_m: float
    waypoint_hold_steps: int

    def __post_init__(self) -> None:
        if (
            self.initial_ground_altitude_m <= 0.0
            or self.construction_altitude_m <= self.initial_ground_altitude_m
            or self.construction_hold_steps < 1
            or self.waypoint_spacing_m <= 0.0
            or self.waypoint_hold_steps < 1
        ):
            raise ConfigurationError(
                "mission altitudes, waypoint spacing, and hold steps must be valid and positive."
            )


@dataclass(frozen=True, slots=True)
class MorphingStudyConfig:
    rate_per_second: float
    release_hold_steps: int
    trigger_clearance_m: float

    def __post_init__(self) -> None:
        if self.rate_per_second <= 0.0 or self.release_hold_steps < 1:
            raise ConfigurationError("morphing rate and release hold must be positive.")
        if self.trigger_clearance_m < 0.0:
            raise ConfigurationError("morphing trigger clearance must be non-negative.")


@dataclass(frozen=True, slots=True)
class RecoveryStudyConfig:
    disturbance_step: int
    velocity_offset: tuple[float, float, float]
    affected_agents: tuple[int, ...]
    error_threshold: float
    threshold_hold_steps: int

    def __post_init__(self) -> None:
        if self.disturbance_step < 1 or not self.affected_agents or min(self.affected_agents) < 0:
            raise ConfigurationError("recovery disturbance selection is invalid.")
        if len(set(self.affected_agents)) != len(self.affected_agents):
            raise ConfigurationError("recovery affected_agents must be unique.")
        if self.error_threshold < 0.0 or self.threshold_hold_steps < 1:
            raise ConfigurationError("recovery threshold and hold are invalid.")


@dataclass(frozen=True, slots=True)
class RecurrentStudyConfig:
    communication: CommunicationExperimentConfig
    recurrent: RecurrentMAPPOConfig
    formation_kinds: tuple[FormationKind, ...]
    mission: MissionStudyConfig
    morphing: MorphingStudyConfig
    recovery: RecoveryStudyConfig

    def __post_init__(self) -> None:
        if self.communication.mappo.algorithm != self.recurrent.mappo:
            raise ConfigurationError("recurrent and physical MAPPO settings must match.")
        if not self.formation_kinds or len(set(self.formation_kinds)) != len(self.formation_kinds):
            raise ConfigurationError("research formation_kinds must be non-empty and unique.")
        formation = self.communication.mappo.experiment.formation
        try:
            for kind in self.formation_kinds:
                equal_size_formation(kind, formation.num_agents, formation.spacing_m)
        except ValueError as error:
            raise ConfigurationError(f"formation pool is incompatible: {error}") from error
        if max(self.recovery.affected_agents) >= formation.num_agents:
            raise ConfigurationError("recovery selects an unavailable UAV index.")


def _mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{path} keys must be strings.")
        result[key] = item
    return result


def _keys(values: Mapping[str, object], *, path: str, expected: set[str]) -> None:
    if set(values) != expected:
        raise ConfigurationError(f"{path} must contain exactly: {', '.join(sorted(expected))}.")


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be numeric.")
    return float(value)


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer.")
    return value


def _list(value: object, *, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a YAML list.")
    return cast(list[object], value)


def recurrent_study_config_from_mapping(value: object) -> RecurrentStudyConfig:
    """Compose communication factors, recurrent architecture, and mission settings."""
    root = _mapping(value, path="configuration")
    research = _mapping(root.get("research"), path="research")
    _keys(
        research,
        path="research",
        expected={"formation_kinds", "mission", "morphing", "recovery"},
    )
    common = {key: item for key, item in root.items() if key not in {"recurrent", "research"}}
    communication = communication_experiment_config_from_mapping(common)
    recurrent = recurrent_settings_from_mapping(
        root.get("recurrent"), communication.mappo.algorithm
    )
    try:
        kinds = tuple(
            FormationKind(item)
            for item in _list(research["formation_kinds"], path="research.formation_kinds")
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError("research.formation_kinds contains an invalid kind.") from error

    mission = _mapping(research["mission"], path="research.mission")
    _keys(
        mission,
        path="research.mission",
        expected={
            "initial_ground_altitude_m",
            "construction_altitude_m",
            "construction_hold_steps",
            "waypoint_spacing_m",
            "waypoint_hold_steps",
        },
    )
    morphing = _mapping(research["morphing"], path="research.morphing")
    _keys(
        morphing,
        path="research.morphing",
        expected={"rate_per_second", "release_hold_steps", "trigger_clearance_m"},
    )
    recovery = _mapping(research["recovery"], path="research.recovery")
    _keys(
        recovery,
        path="research.recovery",
        expected={
            "disturbance_step",
            "velocity_offset",
            "affected_agents",
            "error_threshold",
            "threshold_hold_steps",
        },
    )
    velocity = _list(recovery["velocity_offset"], path="research.recovery.velocity_offset")
    if len(velocity) != 3:
        raise ConfigurationError("research.recovery.velocity_offset must contain three numbers.")
    agents = _list(recovery["affected_agents"], path="research.recovery.affected_agents")
    return RecurrentStudyConfig(
        communication,
        recurrent,
        kinds,
        MissionStudyConfig(
            _number(
                mission["initial_ground_altitude_m"],
                path="research.mission.initial_ground_altitude_m",
            ),
            _number(
                mission["construction_altitude_m"],
                path="research.mission.construction_altitude_m",
            ),
            _integer(
                mission["construction_hold_steps"],
                path="research.mission.construction_hold_steps",
            ),
            _number(
                mission["waypoint_spacing_m"],
                path="research.mission.waypoint_spacing_m",
            ),
            _integer(
                mission["waypoint_hold_steps"],
                path="research.mission.waypoint_hold_steps",
            ),
        ),
        MorphingStudyConfig(
            _number(morphing["rate_per_second"], path="research.morphing.rate_per_second"),
            _integer(morphing["release_hold_steps"], path="research.morphing.release_hold_steps"),
            _number(
                morphing["trigger_clearance_m"],
                path="research.morphing.trigger_clearance_m",
            ),
        ),
        RecoveryStudyConfig(
            _integer(recovery["disturbance_step"], path="research.recovery.disturbance_step"),
            tuple(
                _number(item, path=f"research.recovery.velocity_offset[{index}]")
                for index, item in enumerate(velocity)
            ),  # type: ignore[arg-type]
            tuple(
                _integer(item, path=f"research.recovery.affected_agents[{index}]")
                for index, item in enumerate(agents)
            ),
            _number(recovery["error_threshold"], path="research.recovery.error_threshold"),
            _integer(
                recovery["threshold_hold_steps"],
                path="research.recovery.threshold_hold_steps",
            ),
        ),
    )


def load_recurrent_study_config(path: str | Path) -> RecurrentStudyConfig:
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("recurrent study configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load recurrent study configuration: {error}") from error
    return recurrent_study_config_from_mapping(raw)


__all__ = [
    "MissionStudyConfig",
    "MorphingStudyConfig",
    "RecoveryStudyConfig",
    "RecurrentStudyConfig",
    "load_recurrent_study_config",
    "recurrent_study_config_from_mapping",
]
