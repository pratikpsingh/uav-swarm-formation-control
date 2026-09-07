"""Strict configuration for the neighbor-topology study."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import yaml

from uav_swarm_control.communication import (
    CommunicationCondition,
    CommunicationRegimen,
    CommunicationRegimenKind,
)
from uav_swarm_control.configuration.generalization import (
    AssignmentMode,
    CoordinateFrame,
    GeneralizationVariant,
)
from uav_swarm_control.configuration.mappo import (
    MAPPOExperimentConfig,
    mappo_experiment_config_from_mapping,
)
from uav_swarm_control.configuration.models import ConfigurationError
from uav_swarm_control.configuration.obstacles import (
    formation_pose_range_from_mapping,
    obstacle_field_config_from_mapping,
)
from uav_swarm_control.configuration.pybullet import (
    PyBulletExperimentConfig,
    pybullet_experiment_config_from_mapping,
)
from uav_swarm_control.formations import FormationPoseRange, create_formation
from uav_swarm_control.models import NeighborEncoderSpec
from uav_swarm_control.obstacles import ObstacleFieldConfig, ObstacleScenario
from uav_swarm_control.seeding import validate_seed

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class NeighborEncoderConfig:
    """Architecture and byte-accounting choices held fixed across the study."""

    embedding_size: int
    hidden_sizes: tuple[int, ...]
    payload_bytes_per_neighbor: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.embedding_size, bool)
            or self.embedding_size < 1
            or not self.hidden_sizes
            or any(isinstance(size, bool) or size < 1 for size in self.hidden_sizes)
        ):
            raise ConfigurationError("neighbor encoder sizes must be positive integers.")
        if isinstance(self.payload_bytes_per_neighbor, bool) or self.payload_bytes_per_neighbor < 0:
            raise ConfigurationError("payload_bytes_per_neighbor must be non-negative.")

    def specification(self, max_neighbors: int) -> NeighborEncoderSpec:
        """Bind the common encoder to this task's padded observation capacity."""
        return NeighborEncoderSpec(
            ego_features=6,
            max_neighbors=max_neighbors,
            neighbor_features=6,
            embedding_size=self.embedding_size,
            hidden_sizes=self.hidden_sizes,
        )


@dataclass(frozen=True, slots=True)
class CommunicationExperimentConfig:
    """Physical task, topology factors, training roles, and replication protocol."""

    mappo: MAPPOExperimentConfig
    physics: PyBulletExperimentConfig
    pose: FormationPoseRange
    variant: GeneralizationVariant
    field: ObstacleFieldConfig
    encoder: NeighborEncoderConfig
    evaluation_conditions: tuple[CommunicationCondition, ...]
    training_regimens: tuple[CommunicationRegimen, ...]
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    profile: str

    def __post_init__(self) -> None:
        experiment = self.mappo.experiment
        if experiment != self.physics.experiment:
            raise ConfigurationError("MAPPO and physics must share one experiment configuration.")
        capacity = experiment.formation.num_agents - 1
        if experiment.observation.max_neighbors != capacity:
            raise ConfigurationError(
                "neighbor study observation capacity must include every other agent."
            )
        if experiment.observation.neighbor_radius_m is not None:
            raise ConfigurationError(
                "neighbor study applies sensing range per condition, not in the base observation."
            )
        if not self.evaluation_conditions:
            raise ConfigurationError("neighbor study requires evaluation conditions.")
        names = [condition.name for condition in self.evaluation_conditions]
        factors = [
            (
                condition.requested_neighbors,
                condition.sensing_radius_m,
                condition.obstacle_scenario,
            )
            for condition in self.evaluation_conditions
        ]
        if len(set(names)) != len(names) or len(set(factors)) != len(factors):
            raise ConfigurationError("evaluation conditions must have unique names and factors.")
        if any(
            condition.requested_neighbors > capacity for condition in self.evaluation_conditions
        ):
            raise ConfigurationError("requested neighbors exceed this task's swarm capacity.")
        counts = {condition.requested_neighbors for condition in self.evaluation_conditions}
        radii = {condition.sensing_radius_m for condition in self.evaluation_conditions}
        obstacles = {condition.obstacle_scenario for condition in self.evaluation_conditions}
        if len(counts) < 3 or len(radii) < 2 or len(obstacles) < 2:
            raise ConfigurationError(
                "neighbor study must vary neighbor count, sensing range, and obstacle condition."
            )
        regimen_names = [regimen.name for regimen in self.training_regimens]
        fixed = [
            regimen
            for regimen in self.training_regimens
            if regimen.kind is CommunicationRegimenKind.FIXED
        ]
        variable = [
            regimen
            for regimen in self.training_regimens
            if regimen.kind is CommunicationRegimenKind.VARIABLE
        ]
        if len(set(regimen_names)) != len(regimen_names) or len(fixed) < 2 or len(variable) != 1:
            raise ConfigurationError(
                "Neighbor study requires unique names, two fixed regimens, and one variable."
            )
        known = set(self.evaluation_conditions)
        if any(
            condition not in known
            for regimen in self.training_regimens
            for condition in regimen.conditions
        ):
            raise ConfigurationError("training regimens must reference evaluation conditions.")
        if set(variable[0].conditions) != known:
            raise ConfigurationError(
                "the variable regimen must train over every evaluation condition."
            )
        if len({regimen.conditions[0].requested_neighbors for regimen in fixed}) != len(fixed):
            raise ConfigurationError("fixed regimens must use distinct requested neighbor counts.")
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
        if experiment.reward.smoothness_weight or experiment.reward.success_bonus:
            raise ConfigurationError("neighbor study retains the baseline reward definition.")
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


def _integer(value: object, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{path} must be an integer >= {minimum}.")
    return value


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{path} must be finite.")
    return result


def _condition(value: object, *, index: int) -> CommunicationCondition:
    path = f"communication_study.evaluation_conditions[{index}]"
    values = _mapping(value, path=path)
    _keys(
        values,
        path=path,
        required={"name", "requested_neighbors", "sensing_radius_m", "obstacle_scenario"},
    )
    radius = values["sensing_radius_m"]
    try:
        return CommunicationCondition(
            cast(str, values["name"]),
            _integer(values["requested_neighbors"], path=f"{path}.requested_neighbors"),
            None if radius is None else _number(radius, path=f"{path}.sensing_radius_m"),
            ObstacleScenario(cast(str, values["obstacle_scenario"])),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}: {error}") from error


def _regimen(
    value: object,
    *,
    index: int,
    conditions_by_name: Mapping[str, CommunicationCondition],
) -> CommunicationRegimen:
    path = f"communication_study.training_regimens[{index}]"
    values = _mapping(value, path=path)
    _keys(values, path=path, required={"name", "kind", "conditions"})
    raw_conditions = values["conditions"]
    if not isinstance(raw_conditions, list) or not all(
        isinstance(name, str) for name in cast(list[object], raw_conditions)
    ):
        raise ConfigurationError(f"{path}.conditions must be a YAML list of names.")
    names = cast(list[str], raw_conditions)
    try:
        selected = tuple(conditions_by_name[name] for name in names)
    except KeyError as error:
        raise ConfigurationError(
            f"{path} references unknown condition {error.args[0]!r}."
        ) from error
    try:
        return CommunicationRegimen(
            cast(str, values["name"]),
            CommunicationRegimenKind(cast(str, values["kind"])),
            selected,
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}: {error}") from error


def communication_experiment_config_from_mapping(value: object) -> CommunicationExperimentConfig:
    """Compose the common MAPPO/PyBullet schemas with topology factors."""
    root = _mapping(value, path="configuration")
    study = _mapping(root.get("communication_study"), path="communication_study")
    _keys(
        study,
        path="communication_study",
        required={
            "pose",
            "variant",
            "field",
            "encoder",
            "evaluation_conditions",
            "training_regimens",
        },
    )
    variant_values = _mapping(study["variant"], path="communication_study.variant")
    _keys(
        variant_values,
        path="communication_study.variant",
        required={"name", "assignment", "coordinate_frame"},
    )
    try:
        variant = GeneralizationVariant(
            cast(str, variant_values["name"]),
            AssignmentMode(cast(str, variant_values["assignment"])),
            CoordinateFrame(cast(str, variant_values["coordinate_frame"])),
        )
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "communication_study.variant contains an invalid value."
        ) from error
    encoder_values = _mapping(study["encoder"], path="communication_study.encoder")
    _keys(
        encoder_values,
        path="communication_study.encoder",
        required={"embedding_size", "hidden_sizes", "payload_bytes_per_neighbor"},
    )
    raw_hidden = encoder_values["hidden_sizes"]
    if not isinstance(raw_hidden, list):
        raise ConfigurationError("communication_study.encoder.hidden_sizes must be a YAML list.")
    encoder = NeighborEncoderConfig(
        _integer(
            encoder_values["embedding_size"],
            path="communication_study.encoder.embedding_size",
            minimum=1,
        ),
        tuple(
            _integer(item, path=f"communication_study.encoder.hidden_sizes[{index}]", minimum=1)
            for index, item in enumerate(cast(list[object], raw_hidden))
        ),
        _integer(
            encoder_values["payload_bytes_per_neighbor"],
            path="communication_study.encoder.payload_bytes_per_neighbor",
        ),
    )
    raw_conditions = study["evaluation_conditions"]
    raw_regimens = study["training_regimens"]
    if not isinstance(raw_conditions, list) or not isinstance(raw_regimens, list):
        raise ConfigurationError("evaluation_conditions and training_regimens must be YAML lists.")
    conditions = tuple(
        _condition(item, index=index)
        for index, item in enumerate(cast(list[object], raw_conditions))
    )
    by_name = {condition.name: condition for condition in conditions}
    regimens = tuple(
        _regimen(item, index=index, conditions_by_name=by_name)
        for index, item in enumerate(cast(list[object], raw_regimens))
    )
    protocol = _mapping(root.get("protocol"), path="protocol")
    _keys(protocol, path="protocol", required={"training_seeds", "evaluation_seed", "profile"})
    raw_seeds = protocol["training_seeds"]
    if not isinstance(raw_seeds, list):
        raise ConfigurationError("protocol.training_seeds must be a YAML list.")
    common = {
        key: item for key, item in root.items() if key not in {"communication_study", "protocol"}
    }
    physics_values = {
        key: item for key, item in common.items() if key not in {"algorithm", "evaluation"}
    }
    mappo_values = {key: item for key, item in common.items() if key != "simulator"}
    return CommunicationExperimentConfig(
        mappo_experiment_config_from_mapping(mappo_values),
        pybullet_experiment_config_from_mapping(physics_values),
        formation_pose_range_from_mapping(study["pose"], path="communication_study.pose"),
        variant,
        obstacle_field_config_from_mapping(study["field"], path="communication_study.field"),
        encoder,
        conditions,
        regimens,
        tuple(cast(list[int], raw_seeds)),
        cast(int, protocol["evaluation_seed"]),
        cast(str, protocol["profile"]),
    )


def load_communication_experiment_config(path: str | Path) -> CommunicationExperimentConfig:
    """Safely load one self-contained neighbor study experiment."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("communication configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load communication configuration: {error}") from error
    return communication_experiment_config_from_mapping(raw)


__all__ = [
    "CommunicationExperimentConfig",
    "NeighborEncoderConfig",
    "communication_experiment_config_from_mapping",
    "load_communication_experiment_config",
]
