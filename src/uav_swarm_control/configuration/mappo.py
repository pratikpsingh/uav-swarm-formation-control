"""Validated composition of kinematic-task and MAPPO settings."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.loading import experiment_config_from_mapping
from uav_swarm_control.configuration.models import ConfigurationError, ExperimentConfig
from uav_swarm_control.configuration.ppo import (
    PPO_ALGORITHM_KEYS,
    PPOConfig,
    ppo_config_from_mapping,
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


def _positive_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer.")
    if value < 1:
        raise ConfigurationError(f"{path} must be at least 1.")
    return value


def _hidden_sizes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigurationError("algorithm.critic_hidden_sizes must be a YAML list.")
    sizes = cast(list[object], value)
    if not sizes:
        raise ConfigurationError("algorithm.critic_hidden_sizes must not be empty.")
    return tuple(
        _positive_integer(size, path=f"algorithm.critic_hidden_sizes[{index}]")
        for index, size in enumerate(sizes)
    )


@dataclass(frozen=True, slots=True)
class MAPPOConfig:
    """PPO hyperparameters plus parallel-environment and critic settings."""

    ppo: PPOConfig
    num_environments: int
    critic_hidden_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        environments = _positive_integer(
            self.num_environments,
            path="algorithm.num_environments",
        )
        steps_per_update = self.ppo.rollout_steps * environments
        if self.ppo.total_steps % steps_per_update != 0:
            raise ConfigurationError(
                "algorithm.total_steps must be divisible by rollout_steps * num_environments."
            )
        if not self.critic_hidden_sizes or any(size < 1 for size in self.critic_hidden_sizes):
            raise ConfigurationError(
                "algorithm.critic_hidden_sizes must contain positive integers."
            )
        object.__setattr__(self, "num_environments", environments)

    @property
    def num_updates(self) -> int:
        """Number of parallel collect-update cycles."""
        return self.ppo.total_steps // (self.ppo.rollout_steps * self.num_environments)


@dataclass(frozen=True, slots=True)
class MAPPOExperimentConfig:
    """Complete MAPPO run using the existing kinematic task schema."""

    experiment: ExperimentConfig
    algorithm: MAPPOConfig
    evaluation_episodes: int

    def __post_init__(self) -> None:
        episodes = _positive_integer(self.evaluation_episodes, path="evaluation.episodes")
        samples_per_rollout = (
            self.algorithm.ppo.rollout_steps
            * self.algorithm.num_environments
            * self.experiment.formation.num_agents
        )
        if samples_per_rollout % self.algorithm.ppo.minibatch_size != 0:
            raise ConfigurationError(
                "agent samples per rollout must be divisible by algorithm.minibatch_size."
            )
        object.__setattr__(self, "evaluation_episodes", episodes)


def mappo_experiment_config_from_mapping(value: object) -> MAPPOExperimentConfig:
    """Build a strict MAPPO experiment while reusing the kinematic task schema."""
    root = _mapping(value, path="configuration")
    if "algorithm" not in root or "evaluation" not in root:
        raise ConfigurationError("configuration requires algorithm and evaluation mappings.")

    raw_algorithm = _mapping(root["algorithm"], path="algorithm")
    expected_algorithm_keys = set(PPO_ALGORITHM_KEYS) | {
        "num_environments",
        "critic_hidden_sizes",
    }
    missing = expected_algorithm_keys - raw_algorithm.keys()
    unknown = raw_algorithm.keys() - expected_algorithm_keys
    if missing:
        raise ConfigurationError(
            f"algorithm is missing required keys: {', '.join(sorted(missing))}."
        )
    if unknown:
        raise ConfigurationError(f"algorithm contains unknown keys: {', '.join(sorted(unknown))}.")
    ppo_values = {key: raw_algorithm[key] for key in PPO_ALGORITHM_KEYS}
    algorithm = MAPPOConfig(
        ppo=ppo_config_from_mapping(ppo_values),
        num_environments=_positive_integer(
            raw_algorithm["num_environments"],
            path="algorithm.num_environments",
        ),
        critic_hidden_sizes=_hidden_sizes(raw_algorithm["critic_hidden_sizes"]),
    )

    evaluation = _mapping(root["evaluation"], path="evaluation")
    if set(evaluation) != {"episodes"}:
        raise ConfigurationError("evaluation must contain exactly the episodes key.")

    environment_values = dict(root)
    del environment_values["algorithm"]
    del environment_values["evaluation"]
    experiment = experiment_config_from_mapping(environment_values)
    return MAPPOExperimentConfig(
        experiment=experiment,
        algorithm=algorithm,
        evaluation_episodes=_positive_integer(
            evaluation["episodes"],
            path="evaluation.episodes",
        ),
    )


def load_mappo_experiment_config(path: str | Path) -> MAPPOExperimentConfig:
    """Safely load one self-contained MAPPO YAML experiment."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("MAPPO configuration must use a .yaml or .yml extension.")
    try:
        contents = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"could not read configuration {config_path}: {error}") from error
    try:
        raw = cast(object, yaml.safe_load(contents))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {config_path}: {error}") from error
    return mappo_experiment_config_from_mapping(raw)
