"""Strict configuration for Stage 9 three-dimensional generalization studies."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import yaml

from uav_swarm_control.configuration.mappo import (
    MAPPOExperimentConfig,
    mappo_experiment_config_from_mapping,
)
from uav_swarm_control.configuration.models import ConfigurationError, Vector3
from uav_swarm_control.configuration.pybullet import (
    PyBulletExperimentConfig,
    pybullet_experiment_config_from_mapping,
)
from uav_swarm_control.formations import (
    FormationKind,
    FormationPoseRange,
    create_formation,
    pose_ranges_overlap,
    rotation_matrix_from_euler,
)
from uav_swarm_control.seeding import validate_seed

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SPATIAL_FORMATIONS = {
    FormationKind.PLANE,
    FormationKind.PYRAMID,
    FormationKind.CUBE,
    FormationKind.SPHERE,
}


class AssignmentMode(StrEnum):
    """How stable agent rows are mapped to transformed target points."""

    FIXED = "fixed"
    MINIMUM_DISTANCE = "minimum-distance"


class CoordinateFrame(StrEnum):
    """Frame used by the actor observation and decoded velocity action."""

    WORLD = "world"
    TARGET = "target"


@dataclass(frozen=True, slots=True)
class GeneralizationVariant:
    """One assignment/frame ablation under otherwise identical conditions."""

    name: str
    assignment: AssignmentMode
    coordinate_frame: CoordinateFrame

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ConfigurationError("variant.name must use lowercase kebab-case.")
        try:
            assignment = AssignmentMode(self.assignment)
            frame = CoordinateFrame(self.coordinate_frame)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                "variant assignment or coordinate frame is invalid."
            ) from error
        object.__setattr__(self, "assignment", assignment)
        object.__setattr__(self, "coordinate_frame", frame)


@dataclass(frozen=True, slots=True)
class GeneralizationConfig:
    """Physical MAPPO task, disjoint pose splits, variants and seed protocol."""

    mappo: MAPPOExperimentConfig
    physics: PyBulletExperimentConfig
    training_pose: FormationPoseRange
    held_out_pose: FormationPoseRange
    variants: tuple[GeneralizationVariant, ...]
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    profile: str

    def __post_init__(self) -> None:
        experiment = self.mappo.experiment
        if experiment != self.physics.experiment:
            raise ConfigurationError("MAPPO and physics must share one experiment configuration.")
        if experiment.formation.kind not in _SPATIAL_FORMATIONS:
            choices = ", ".join(sorted(item.value for item in _SPATIAL_FORMATIONS))
            raise ConfigurationError(f"Stage 9 formation.kind must be one of: {choices}.")
        if not self.variants:
            raise ConfigurationError("generalization requires at least one variant.")
        names = [variant.name for variant in self.variants]
        choices = [(variant.assignment, variant.coordinate_frame) for variant in self.variants]
        if len(set(names)) != len(names) or len(set(choices)) != len(choices):
            raise ConfigurationError("generalization variants require unique names and choices.")
        if pose_ranges_overlap(self.training_pose, self.held_out_pose):
            raise ConfigurationError("training and held-out pose ranges must be disjoint.")
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
            raise ConfigurationError("3D generalization requires an active formation reward.")
        if experiment.reward.smoothness_weight or experiment.reward.success_bonus:
            raise ConfigurationError("Stage 9 retains the corrected Paper 04 reward definition.")

        minimum_scale = min(self.training_pose.scale_lower, self.held_out_pose.scale_lower)
        if minimum_scale * experiment.formation.spacing_m <= experiment.task.collision_distance_m:
            raise ConfigurationError("minimum target spacing must exceed collision distance.")
        template = create_formation(
            experiment.formation.kind,
            experiment.formation.num_agents,
            experiment.formation.spacing_m,
        )
        maximum_scale = max(self.training_pose.scale_upper, self.held_out_pose.scale_upper)
        radius = float(np.linalg.norm(template, axis=1).max()) * maximum_scale
        target_center_z = experiment.formation.center_m[2] + min(
            self.training_pose.center_offset_lower_m[2],
            self.held_out_pose.center_offset_lower_m[2],
        )
        if target_center_z - radius <= 0.0:
            raise ConfigurationError("pose ranges can place a target point at or below the ground.")
        rotation = rotation_matrix_from_euler(*experiment.formation.euler_radians)
        initial_relative = template @ rotation.T
        minimum_initial_z = (
            experiment.task.initial_center_m[2]
            + float(initial_relative[:, 2].min())
            - experiment.task.initial_position_noise_m
        )
        if minimum_initial_z <= 0.0:
            raise ConfigurationError("initial formation can be sampled at or below the ground.")
        object.__setattr__(self, "training_seeds", seeds)
        object.__setattr__(self, "evaluation_seed", evaluation_seed)


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


def _vector(value: object, *, path: str) -> Vector3:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a YAML list containing three numbers.")
    items = cast(list[object], value)
    if len(items) != 3:
        raise ConfigurationError(f"{path} must be a YAML list containing three numbers.")
    return cast(Vector3, tuple(_number(item, path=f"{path}[{i}]") for i, item in enumerate(items)))


def _pose_range(value: object, *, path: str) -> FormationPoseRange:
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
                values["euler_offset_lower_radians"],
                path=f"{path}.euler_offset_lower_radians",
            ),
            _vector(
                values["euler_offset_upper_radians"],
                path=f"{path}.euler_offset_upper_radians",
            ),
            _number(values["scale_lower"], path=f"{path}.scale_lower"),
            _number(values["scale_upper"], path=f"{path}.scale_upper"),
        )
    except ValueError as error:
        raise ConfigurationError(f"{path}: {error}") from error


def _variant(value: object, *, index: int) -> GeneralizationVariant:
    path = f"generalization.variants[{index}]"
    values = _mapping(value, path=path)
    _keys(values, path=path, required={"name", "assignment", "coordinate_frame"})
    try:
        return GeneralizationVariant(
            name=cast(str, values["name"]),
            assignment=AssignmentMode(cast(str, values["assignment"])),
            coordinate_frame=CoordinateFrame(cast(str, values["coordinate_frame"])),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} contains an invalid enum value.") from error


def generalization_config_from_mapping(value: object) -> GeneralizationConfig:
    """Compose existing MAPPO/PyBullet schemas with strict Stage 9 choices."""
    root = _mapping(value, path="configuration")
    generalization = _mapping(root.get("generalization"), path="generalization")
    _keys(
        generalization,
        path="generalization",
        required={"training_pose", "held_out_pose", "variants"},
    )
    raw_variants = generalization["variants"]
    if not isinstance(raw_variants, list):
        raise ConfigurationError("generalization.variants must be a YAML list.")
    variants = cast(list[object], raw_variants)
    protocol = _mapping(root.get("protocol"), path="protocol")
    _keys(
        protocol,
        path="protocol",
        required={"training_seeds", "evaluation_seed", "profile"},
    )
    raw_seeds = protocol["training_seeds"]
    if not isinstance(raw_seeds, list):
        raise ConfigurationError("protocol.training_seeds must be a YAML list.")
    common = {key: item for key, item in root.items() if key not in {"generalization", "protocol"}}
    physics_values = {
        key: item for key, item in common.items() if key not in {"algorithm", "evaluation"}
    }
    mappo_values = {key: item for key, item in common.items() if key != "simulator"}
    return GeneralizationConfig(
        mappo=mappo_experiment_config_from_mapping(mappo_values),
        physics=pybullet_experiment_config_from_mapping(physics_values),
        training_pose=_pose_range(
            generalization["training_pose"], path="generalization.training_pose"
        ),
        held_out_pose=_pose_range(
            generalization["held_out_pose"], path="generalization.held_out_pose"
        ),
        variants=tuple(_variant(item, index=index) for index, item in enumerate(variants)),
        training_seeds=tuple(cast(list[int], raw_seeds)),
        evaluation_seed=cast(int, protocol["evaluation_seed"]),
        profile=cast(str, protocol["profile"]),
    )


def load_generalization_config(path: str | Path) -> GeneralizationConfig:
    """Safely load one self-contained Stage 9 experiment."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("generalization configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load generalization configuration: {error}") from error
    return generalization_config_from_mapping(raw)


__all__ = [
    "AssignmentMode",
    "CoordinateFrame",
    "GeneralizationConfig",
    "GeneralizationVariant",
    "generalization_config_from_mapping",
    "load_generalization_config",
]
