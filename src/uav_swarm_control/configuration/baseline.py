"""Composition of existing MAPPO and PyBullet schemas for reproducible baselines."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.mappo import (
    MAPPOExperimentConfig,
    mappo_experiment_config_from_mapping,
)
from uav_swarm_control.configuration.models import ConfigurationError
from uav_swarm_control.configuration.pybullet import (
    PyBulletExperimentConfig,
    pybullet_experiment_config_from_mapping,
)
from uav_swarm_control.seeding import validate_seed


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """One task and its independent training/evaluation seed protocol."""

    mappo: MAPPOExperimentConfig
    physics: PyBulletExperimentConfig
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    profile: str


def load_baseline_config(path: str | Path) -> BaselineConfig:
    """Load a strict self-contained corrected Paper 04 baseline."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("baseline configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load baseline configuration: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError("baseline configuration must be a string-keyed mapping.")
    if not all(isinstance(key, str) for key in cast(dict[object, object], raw)):
        raise ConfigurationError("baseline configuration must be a string-keyed mapping.")
    root = cast(dict[str, object], raw)
    protocol = root.get("protocol")
    if not isinstance(protocol, dict):
        raise ConfigurationError("protocol must be a mapping.")
    protocol = cast(dict[str, object], protocol)
    if set(protocol) != {"training_seeds", "evaluation_seed", "profile"}:
        raise ConfigurationError("protocol requires training_seeds, evaluation_seed, profile.")
    seed_values = protocol["training_seeds"]
    if not isinstance(seed_values, list):
        raise ConfigurationError("protocol.training_seeds must be a list.")
    try:
        seeds = tuple(validate_seed(value) for value in cast(list[object], seed_values))
        evaluation_seed = validate_seed(protocol["evaluation_seed"])
    except (TypeError, ValueError) as error:
        raise ConfigurationError(str(error)) from error
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ConfigurationError("protocol requires at least five distinct training seeds.")
    profile = protocol["profile"]
    if not isinstance(profile, str) or profile not in {"smoke", "research"}:
        raise ConfigurationError("protocol.profile must be smoke or research.")
    common = {key: value for key, value in root.items() if key != "protocol"}
    physics_values = {
        key: value for key, value in common.items() if key not in {"algorithm", "evaluation"}
    }
    mappo_values = {key: value for key, value in common.items() if key != "simulator"}
    physics = pybullet_experiment_config_from_mapping(physics_values)
    mappo = mappo_experiment_config_from_mapping(mappo_values)
    if mappo.experiment.formation.num_agents not in {3, 4, 5}:
        raise ConfigurationError("Paper 04 baseline requires 3, 4, or 5 agents.")
    if mappo.experiment.reward.formation_weight <= 0:
        raise ConfigurationError("corrected baseline requires an active formation reward.")
    if mappo.experiment.reward.smoothness_weight or mappo.experiment.reward.success_bonus:
        raise ConfigurationError("baseline reward excludes smoothness shaping and success bonus.")
    return BaselineConfig(mappo, physics, seeds, evaluation_seed, profile)
