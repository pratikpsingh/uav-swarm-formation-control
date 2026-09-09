"""Strict configuration for the paper-aligned recurrent MAPPO method."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.mappo import (
    MAPPOConfig,
    MAPPOExperimentConfig,
    mappo_experiment_config_from_mapping,
)
from uav_swarm_control.configuration.models import ConfigurationError


class RecurrentObservationProfile(StrEnum):
    """Supported decentralized actor encodings."""

    PAPER_FLAT = "paper-flat"
    MASKED_SET = "masked-set"


@dataclass(frozen=True, slots=True)
class RecurrentMAPPOConfig:
    """Recurrent architecture and sequence-update settings around shared PPO."""

    mappo: MAPPOConfig
    feature_size: int
    hidden_size: int
    num_layers: int
    sequence_length: int
    sequences_per_minibatch: int
    observation_profile: RecurrentObservationProfile
    direction_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "feature_size",
            "hidden_size",
            "num_layers",
            "sequence_length",
            "sequences_per_minibatch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"recurrent.{name} must be a positive integer.")
        if self.mappo.ppo.rollout_steps % self.sequence_length != 0:
            raise ConfigurationError(
                "recurrent.sequence_length must divide algorithm.rollout_steps."
            )
        sequences = (
            self.mappo.ppo.rollout_steps // self.sequence_length * self.mappo.num_environments
        )
        if sequences % self.sequences_per_minibatch != 0:
            raise ConfigurationError(
                "rollout sequence count must be divisible by sequences_per_minibatch."
            )
        if self.direction_epsilon <= 0.0:
            raise ConfigurationError("recurrent.direction_epsilon must be positive.")
        object.__setattr__(
            self,
            "observation_profile",
            RecurrentObservationProfile(self.observation_profile),
        )


@dataclass(frozen=True, slots=True)
class RecurrentMAPPOExperimentConfig:
    """A complete kinematic recurrent experiment."""

    base: MAPPOExperimentConfig
    recurrent: RecurrentMAPPOConfig

    def __post_init__(self) -> None:
        if self.base.algorithm != self.recurrent.mappo:
            raise ConfigurationError("base and recurrent configuration must share MAPPO settings.")


def _mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{path} keys must be strings.")
        result[key] = item
    return result


def _positive_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{path} must be a positive integer.")
    return value


def recurrent_settings_from_mapping(
    value: object,
    mappo: MAPPOConfig,
) -> RecurrentMAPPOConfig:
    """Parse only the recurrent section for reuse by physical study loaders."""
    values = _mapping(value, path="recurrent")
    required = {
        "feature_size",
        "hidden_size",
        "num_layers",
        "sequence_length",
        "sequences_per_minibatch",
        "observation_profile",
        "action_profile",
    }
    optional = {"direction_epsilon"}
    missing = required - values.keys()
    unknown = values.keys() - required - optional
    if missing:
        raise ConfigurationError(f"recurrent is missing keys: {', '.join(sorted(missing))}.")
    if unknown:
        raise ConfigurationError(f"recurrent contains unknown keys: {', '.join(sorted(unknown))}.")
    if values["action_profile"] != "direction-speed":
        raise ConfigurationError("recurrent.action_profile must be direction-speed.")
    try:
        profile = RecurrentObservationProfile(cast(str, values["observation_profile"]))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "recurrent.observation_profile must be paper-flat or masked-set."
        ) from error
    epsilon = values.get("direction_epsilon", 1e-6)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ConfigurationError("recurrent.direction_epsilon must be a number.")
    return RecurrentMAPPOConfig(
        mappo=mappo,
        feature_size=_positive_integer(values["feature_size"], path="recurrent.feature_size"),
        hidden_size=_positive_integer(values["hidden_size"], path="recurrent.hidden_size"),
        num_layers=_positive_integer(values["num_layers"], path="recurrent.num_layers"),
        sequence_length=_positive_integer(
            values["sequence_length"], path="recurrent.sequence_length"
        ),
        sequences_per_minibatch=_positive_integer(
            values["sequences_per_minibatch"],
            path="recurrent.sequences_per_minibatch",
        ),
        observation_profile=profile,
        direction_epsilon=float(epsilon),
    )


def recurrent_mappo_experiment_config_from_mapping(
    value: object,
) -> RecurrentMAPPOExperimentConfig:
    """Compose the common kinematic MAPPO schema with recurrent settings."""
    root = _mapping(value, path="configuration")
    if "recurrent" not in root:
        raise ConfigurationError("configuration requires a recurrent mapping.")
    base_values = {key: item for key, item in root.items() if key != "recurrent"}
    base = mappo_experiment_config_from_mapping(base_values)
    recurrent = recurrent_settings_from_mapping(root["recurrent"], base.algorithm)
    return RecurrentMAPPOExperimentConfig(base, recurrent)


def load_recurrent_mappo_experiment_config(
    path: str | Path,
) -> RecurrentMAPPOExperimentConfig:
    """Load a paper-aligned recurrent kinematic experiment."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("recurrent MAPPO configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load recurrent MAPPO configuration: {error}") from error
    return recurrent_mappo_experiment_config_from_mapping(raw)


__all__ = [
    "RecurrentMAPPOConfig",
    "RecurrentMAPPOExperimentConfig",
    "RecurrentObservationProfile",
    "load_recurrent_mappo_experiment_config",
    "recurrent_mappo_experiment_config_from_mapping",
    "recurrent_settings_from_mapping",
]
