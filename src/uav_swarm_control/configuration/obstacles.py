"""Strict configuration for the Stage 10 oracle-obstacle study."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import yaml

from uav_swarm_control.configuration.generalization import (
    AssignmentMode,
    CoordinateFrame,
    GeneralizationVariant,
)
from uav_swarm_control.configuration.mappo import (
    MAPPOExperimentConfig,
    mappo_experiment_config_from_mapping,
)
from uav_swarm_control.configuration.models import ConfigurationError, Vector3
from uav_swarm_control.configuration.pybullet import (
    PyBulletExperimentConfig,
    pybullet_experiment_config_from_mapping,
)
from uav_swarm_control.formations import FormationKind, FormationPoseRange, create_formation
from uav_swarm_control.obstacles import ObstacleFieldConfig, ObstacleScenario
from uav_swarm_control.seeding import validate_seed

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class TrainingRole(StrEnum):
    """Experimental role of a training distribution."""

    NO_OBSTACLE_CONTROL = "no-obstacle-control"
    STATIC_CONTROL = "static-control"
    CURRICULUM = "curriculum"


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    """Use ``scenario`` until this fraction of per-environment training."""

    end_fraction: float
    scenario: ObstacleScenario

    def __post_init__(self) -> None:
        if not math.isfinite(self.end_fraction) or not 0.0 < self.end_fraction <= 1.0:
            raise ConfigurationError("curriculum end_fraction must lie in (0, 1].")
        try:
            scenario = ObstacleScenario(self.scenario)
        except (TypeError, ValueError) as error:
            raise ConfigurationError("curriculum scenario is invalid.") from error
        object.__setattr__(self, "scenario", scenario)


@dataclass(frozen=True, slots=True)
class ObstacleTrainingRegimen:
    """Named training distribution used in one controlled comparison."""

    name: str
    role: TrainingRole
    phases: tuple[CurriculumPhase, ...]

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ConfigurationError("training regimen name must use lowercase kebab-case.")
        try:
            role = TrainingRole(self.role)
        except (TypeError, ValueError) as error:
            raise ConfigurationError("training regimen role is invalid.") from error
        if not self.phases:
            raise ConfigurationError("training regimen requires at least one phase.")
        ends = [phase.end_fraction for phase in self.phases]
        if any(right <= left for left, right in pairwise(ends)):
            raise ConfigurationError("curriculum phase endpoints must be strictly increasing.")
        if not math.isclose(ends[-1], 1.0):
            raise ConfigurationError("the final curriculum phase must end at 1.0.")
        expected = {
            TrainingRole.NO_OBSTACLE_CONTROL: (ObstacleScenario.NONE,),
            TrainingRole.STATIC_CONTROL: (ObstacleScenario.STATIC,),
            TrainingRole.CURRICULUM: (
                ObstacleScenario.NONE,
                ObstacleScenario.STATIC,
                ObstacleScenario.SLOW_DYNAMIC,
                ObstacleScenario.MIXED_DYNAMIC,
            ),
        }[role]
        if tuple(phase.scenario for phase in self.phases) != expected:
            raise ConfigurationError(
                f"{role.value} requires scenarios {[item.value for item in expected]}."
            )
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class ObstacleExperimentConfig:
    """Physical MAPPO task plus matched obstacle controls and seed protocol."""

    mappo: MAPPOExperimentConfig
    physics: PyBulletExperimentConfig
    pose: FormationPoseRange
    variant: GeneralizationVariant
    field: ObstacleFieldConfig
    training_regimens: tuple[ObstacleTrainingRegimen, ...]
    evaluation_scenarios: tuple[ObstacleScenario, ...]
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    profile: str

    def __post_init__(self) -> None:
        experiment = self.mappo.experiment
        if experiment != self.physics.experiment:
            raise ConfigurationError("MAPPO and physics must share one experiment configuration.")
        if experiment.formation.kind is not FormationKind.PLANE:
            raise ConfigurationError("Stage 10 fixes formation.kind to plane to isolate obstacles.")
        if experiment.observation.max_neighbors != experiment.formation.num_agents - 1:
            raise ConfigurationError(
                "Stage 10 fixes sensing to every other agent; neighbors vary later."
            )
        roles = [regimen.role for regimen in self.training_regimens]
        names = [regimen.name for regimen in self.training_regimens]
        if set(roles) != set(TrainingRole) or len(roles) != len(TrainingRole):
            raise ConfigurationError(
                "Stage 10 requires exactly one regimen for each training role."
            )
        if len(set(names)) != len(names):
            raise ConfigurationError("training regimen names must be unique.")
        required_scenarios = tuple(ObstacleScenario)
        if self.evaluation_scenarios != required_scenarios:
            raise ConfigurationError(
                "evaluation_scenarios must list each Stage 10 scenario in order."
            )
        if len(self.training_seeds) < 5 or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ConfigurationError("protocol requires at least five distinct training seeds.")
        try:
            seeds = tuple(validate_seed(seed) for seed in self.training_seeds)
            evaluation_seed = validate_seed(self.evaluation_seed)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error
        if self.profile not in {"smoke", "research"}:
            raise ConfigurationError("protocol.profile must be smoke or research.")
        if experiment.reward.formation_weight <= 0.0:
            raise ConfigurationError("obstacle avoidance requires an active formation reward.")
        if experiment.reward.smoothness_weight or experiment.reward.success_bonus:
            raise ConfigurationError("Stage 10 retains the corrected Paper 04 reward definition.")
        if self.pose.scale_lower * experiment.formation.spacing_m <= (
            experiment.task.collision_distance_m
        ):
            raise ConfigurationError("minimum target spacing must exceed collision distance.")
        template = create_formation(
            experiment.formation.kind,
            experiment.formation.num_agents,
            experiment.formation.spacing_m,
        )
        radius = float(np.linalg.norm(template, axis=1).max()) * self.pose.scale_upper
        if experiment.formation.center_m[2] + self.pose.center_offset_lower_m[2] - radius <= 0.0:
            raise ConfigurationError("pose range can place a target point at or below ground.")
        object.__setattr__(self, "training_seeds", seeds)
        object.__setattr__(self, "evaluation_seed", evaluation_seed)


def scenario_for_progress(phases: tuple[CurriculumPhase, ...], progress: float) -> ObstacleScenario:
    """Select the first phase whose inclusive endpoint contains progress."""
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("training progress must lie in [0, 1].")
    return next(
        (phase.scenario for phase in phases if progress <= phase.end_fraction), phases[-1].scenario
    )


def _mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{path} keys must be strings.")
        result[key] = item
    return result


def _keys(values: Mapping[str, object], *, path: str, required: set[str]) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required
    if missing:
        raise ConfigurationError(f"{path} is missing required keys: {', '.join(sorted(missing))}.")
    if unknown:
        raise ConfigurationError(f"{path} contains unknown keys: {', '.join(sorted(unknown))}.")


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{path} must be finite.")
    return result


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer.")
    return value


def _vector(value: object, *, path: str) -> Vector3:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a YAML list containing three numbers.")
    items = cast(list[object], value)
    if len(items) != 3:
        raise ConfigurationError(f"{path} must be a YAML list containing three numbers.")
    return cast(Vector3, tuple(_number(item, path=f"{path}[{i}]") for i, item in enumerate(items)))


def formation_pose_range_from_mapping(
    value: object, *, path: str = "obstacle_study.pose"
) -> FormationPoseRange:
    """Parse a validated formation pose range for obstacle-aware studies."""
    values = _mapping(value, path=path)
    required = {
        "center_offset_lower_m",
        "center_offset_upper_m",
        "euler_offset_lower_radians",
        "euler_offset_upper_radians",
        "scale_lower",
        "scale_upper",
    }
    _keys(values, path=path, required=required)
    try:
        return FormationPoseRange(
            _vector(values["center_offset_lower_m"], path=f"{path}.center_offset_lower_m"),
            _vector(values["center_offset_upper_m"], path=f"{path}.center_offset_upper_m"),
            _vector(
                values["euler_offset_lower_radians"], path=f"{path}.euler_offset_lower_radians"
            ),
            _vector(
                values["euler_offset_upper_radians"], path=f"{path}.euler_offset_upper_radians"
            ),
            _number(values["scale_lower"], path=f"{path}.scale_lower"),
            _number(values["scale_upper"], path=f"{path}.scale_upper"),
        )
    except ValueError as error:
        raise ConfigurationError(f"{path}: {error}") from error


def obstacle_field_config_from_mapping(
    value: object, *, path: str = "obstacle_study.field"
) -> ObstacleFieldConfig:
    """Parse a validated spherical-obstacle field."""
    values = _mapping(value, path=path)
    names = set(ObstacleFieldConfig.__dataclass_fields__)
    _keys(values, path=path, required=names)
    observation_radius = values["observation_radius_m"]
    try:
        return ObstacleFieldConfig(
            max_obstacles=_integer(values["max_obstacles"], path=f"{path}.max_obstacles"),
            static_count=_integer(values["static_count"], path=f"{path}.static_count"),
            slow_dynamic_count=_integer(
                values["slow_dynamic_count"], path=f"{path}.slow_dynamic_count"
            ),
            mixed_count=_integer(values["mixed_count"], path=f"{path}.mixed_count"),
            vehicle_radius_m=_number(values["vehicle_radius_m"], path=f"{path}.vehicle_radius_m"),
            radius_lower_m=_number(values["radius_lower_m"], path=f"{path}.radius_lower_m"),
            radius_upper_m=_number(values["radius_upper_m"], path=f"{path}.radius_upper_m"),
            speed_lower_mps=_number(values["speed_lower_mps"], path=f"{path}.speed_lower_mps"),
            speed_upper_mps=_number(values["speed_upper_mps"], path=f"{path}.speed_upper_mps"),
            route_fraction_lower=_number(
                values["route_fraction_lower"], path=f"{path}.route_fraction_lower"
            ),
            route_fraction_upper=_number(
                values["route_fraction_upper"], path=f"{path}.route_fraction_upper"
            ),
            static_lateral_lower_m=_number(
                values["static_lateral_lower_m"], path=f"{path}.static_lateral_lower_m"
            ),
            static_lateral_upper_m=_number(
                values["static_lateral_upper_m"], path=f"{path}.static_lateral_upper_m"
            ),
            dynamic_lateral_lower_m=_number(
                values["dynamic_lateral_lower_m"], path=f"{path}.dynamic_lateral_lower_m"
            ),
            dynamic_lateral_upper_m=_number(
                values["dynamic_lateral_upper_m"], path=f"{path}.dynamic_lateral_upper_m"
            ),
            vertical_offset_lower_m=_number(
                values["vertical_offset_lower_m"], path=f"{path}.vertical_offset_lower_m"
            ),
            vertical_offset_upper_m=_number(
                values["vertical_offset_upper_m"], path=f"{path}.vertical_offset_upper_m"
            ),
            mixed_dynamic_probability=_number(
                values["mixed_dynamic_probability"], path=f"{path}.mixed_dynamic_probability"
            ),
            observation_radius_m=(
                None
                if observation_radius is None
                else _number(observation_radius, path=f"{path}.observation_radius_m")
            ),
            safety_margin_m=_number(values["safety_margin_m"], path=f"{path}.safety_margin_m"),
            proximity_weight=_number(values["proximity_weight"], path=f"{path}.proximity_weight"),
            collision_penalty=_number(
                values["collision_penalty"], path=f"{path}.collision_penalty"
            ),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}: {error}") from error


def _phase(value: object, *, regimen_index: int, phase_index: int) -> CurriculumPhase:
    path = f"obstacle_study.training_regimens[{regimen_index}].phases[{phase_index}]"
    values = _mapping(value, path=path)
    _keys(values, path=path, required={"end_fraction", "scenario"})
    try:
        return CurriculumPhase(
            _number(values["end_fraction"], path=f"{path}.end_fraction"),
            ObstacleScenario(cast(str, values["scenario"])),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} contains an invalid value.") from error


def _regimen(value: object, *, index: int) -> ObstacleTrainingRegimen:
    path = f"obstacle_study.training_regimens[{index}]"
    values = _mapping(value, path=path)
    _keys(values, path=path, required={"name", "role", "phases"})
    raw_phases = values["phases"]
    if not isinstance(raw_phases, list):
        raise ConfigurationError(f"{path}.phases must be a YAML list.")
    phase_values = cast(list[object], raw_phases)
    try:
        role = TrainingRole(cast(str, values["role"]))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}.role is invalid.") from error
    return ObstacleTrainingRegimen(
        cast(str, values["name"]),
        role,
        tuple(
            _phase(item, regimen_index=index, phase_index=i) for i, item in enumerate(phase_values)
        ),
    )


def obstacle_experiment_config_from_mapping(value: object) -> ObstacleExperimentConfig:
    """Compose existing MAPPO/PyBullet schemas with strict obstacle choices."""
    root = _mapping(value, path="configuration")
    study = _mapping(root.get("obstacle_study"), path="obstacle_study")
    _keys(
        study,
        path="obstacle_study",
        required={"pose", "variant", "field", "training_regimens", "evaluation_scenarios"},
    )
    variant_values = _mapping(study["variant"], path="obstacle_study.variant")
    _keys(
        variant_values,
        path="obstacle_study.variant",
        required={"name", "assignment", "coordinate_frame"},
    )
    try:
        variant = GeneralizationVariant(
            cast(str, variant_values["name"]),
            AssignmentMode(cast(str, variant_values["assignment"])),
            CoordinateFrame(cast(str, variant_values["coordinate_frame"])),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError("obstacle_study.variant contains an invalid value.") from error
    raw_regimens = study["training_regimens"]
    raw_scenarios = study["evaluation_scenarios"]
    if not isinstance(raw_regimens, list) or not isinstance(raw_scenarios, list):
        raise ConfigurationError("training_regimens and evaluation_scenarios must be YAML lists.")
    regimen_values = cast(list[object], raw_regimens)
    scenario_values = cast(list[object], raw_scenarios)
    protocol = _mapping(root.get("protocol"), path="protocol")
    _keys(protocol, path="protocol", required={"training_seeds", "evaluation_seed", "profile"})
    raw_seeds = protocol["training_seeds"]
    if not isinstance(raw_seeds, list):
        raise ConfigurationError("protocol.training_seeds must be a YAML list.")
    common = {key: item for key, item in root.items() if key not in {"obstacle_study", "protocol"}}
    physics_values = {
        key: item for key, item in common.items() if key not in {"algorithm", "evaluation"}
    }
    mappo_values = {key: item for key, item in common.items() if key != "simulator"}
    try:
        scenarios = tuple(ObstacleScenario(cast(str, item)) for item in scenario_values)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("evaluation_scenarios contains an invalid scenario.") from error
    return ObstacleExperimentConfig(
        mappo_experiment_config_from_mapping(mappo_values),
        pybullet_experiment_config_from_mapping(physics_values),
        formation_pose_range_from_mapping(study["pose"]),
        variant,
        obstacle_field_config_from_mapping(study["field"]),
        tuple(_regimen(item, index=i) for i, item in enumerate(regimen_values)),
        scenarios,
        tuple(cast(list[int], raw_seeds)),
        cast(int, protocol["evaluation_seed"]),
        cast(str, protocol["profile"]),
    )


def load_obstacle_experiment_config(path: str | Path) -> ObstacleExperimentConfig:
    """Safely load one self-contained Stage 10 experiment."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("obstacle configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load obstacle configuration: {error}") from error
    return obstacle_experiment_config_from_mapping(raw)


__all__ = [
    "CurriculumPhase",
    "ObstacleExperimentConfig",
    "ObstacleTrainingRegimen",
    "TrainingRole",
    "formation_pose_range_from_mapping",
    "load_obstacle_experiment_config",
    "obstacle_experiment_config_from_mapping",
    "obstacle_field_config_from_mapping",
    "scenario_for_progress",
]
