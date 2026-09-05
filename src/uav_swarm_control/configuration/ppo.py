"""Validated configuration for the single-agent PPO reference experiment."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from uav_swarm_control.configuration.models import ConfigurationError
from uav_swarm_control.seeding import validate_seed

type DeviceName = Literal["auto", "cpu", "cuda"]

_EXPERIMENT_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PPO_ALGORITHM_KEYS = frozenset(
    {
        "total_steps",
        "rollout_steps",
        "update_epochs",
        "minibatch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_coefficient",
        "value_coefficient",
        "entropy_coefficient",
        "max_gradient_norm",
        "hidden_sizes",
        "initial_log_standard_deviation",
        "device",
    }
)


def _integer(value: object, *, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer.")
    if value < minimum:
        raise ConfigurationError(f"{path} must be at least {minimum}.")
    return value


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{path} must be finite.")
    return result


def _positive_number(value: object, *, path: str) -> float:
    result = _number(value, path=path)
    if result <= 0.0:
        raise ConfigurationError(f"{path} must be greater than zero.")
    return result


def _probability(
    value: object,
    *,
    path: str,
    include_zero: bool = True,
    include_one: bool = True,
) -> float:
    result = _number(value, path=path)
    lower_valid = result >= 0.0 if include_zero else result > 0.0
    upper_valid = result <= 1.0 if include_one else result < 1.0
    if not lower_valid or not upper_valid:
        left = "[" if include_zero else "("
        right = "]" if include_one else ")"
        boundary = f"{left}0, 1{right}"
        raise ConfigurationError(f"{path} must be in {boundary}.")
    return result


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{path} must be a string.")
    return value


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
) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required
    if missing:
        raise ConfigurationError(f"{path} is missing required keys: {', '.join(sorted(missing))}.")
    if unknown:
        raise ConfigurationError(f"{path} contains unknown keys: {', '.join(sorted(unknown))}.")


def _hidden_sizes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigurationError("algorithm.hidden_sizes must be a YAML list.")
    items = cast(list[object], value)
    if not items:
        raise ConfigurationError("algorithm.hidden_sizes must not be empty.")
    return tuple(
        _integer(item, path=f"algorithm.hidden_sizes[{index}]") for index, item in enumerate(items)
    )


def _device_name(value: object) -> DeviceName:
    name = _string(value, path="algorithm.device")
    if name not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("algorithm.device must be one of: auto, cpu, cuda.")
    return cast(DeviceName, name)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters for one reproducible PPO training run."""

    total_steps: int
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coefficient: float
    value_coefficient: float
    entropy_coefficient: float
    max_gradient_norm: float
    hidden_sizes: tuple[int, ...]
    initial_log_standard_deviation: float
    device: DeviceName

    def __post_init__(self) -> None:
        total_steps = _integer(self.total_steps, path="algorithm.total_steps")
        rollout_steps = _integer(self.rollout_steps, path="algorithm.rollout_steps")
        update_epochs = _integer(self.update_epochs, path="algorithm.update_epochs")
        minibatch_size = _integer(self.minibatch_size, path="algorithm.minibatch_size")
        if total_steps % rollout_steps != 0:
            raise ConfigurationError("algorithm.total_steps must be divisible by rollout_steps.")
        if rollout_steps % minibatch_size != 0:
            raise ConfigurationError("algorithm.rollout_steps must be divisible by minibatch_size.")
        object.__setattr__(self, "total_steps", total_steps)
        object.__setattr__(self, "rollout_steps", rollout_steps)
        object.__setattr__(self, "update_epochs", update_epochs)
        object.__setattr__(self, "minibatch_size", minibatch_size)
        object.__setattr__(
            self,
            "learning_rate",
            _positive_number(self.learning_rate, path="algorithm.learning_rate"),
        )
        object.__setattr__(self, "gamma", _probability(self.gamma, path="algorithm.gamma"))
        object.__setattr__(
            self,
            "gae_lambda",
            _probability(self.gae_lambda, path="algorithm.gae_lambda"),
        )
        object.__setattr__(
            self,
            "clip_coefficient",
            _probability(
                self.clip_coefficient,
                path="algorithm.clip_coefficient",
                include_zero=False,
                include_one=False,
            ),
        )
        object.__setattr__(
            self,
            "value_coefficient",
            _number(self.value_coefficient, path="algorithm.value_coefficient"),
        )
        object.__setattr__(
            self,
            "entropy_coefficient",
            _number(self.entropy_coefficient, path="algorithm.entropy_coefficient"),
        )
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ConfigurationError("loss coefficients must be non-negative.")
        object.__setattr__(
            self,
            "max_gradient_norm",
            _positive_number(self.max_gradient_norm, path="algorithm.max_gradient_norm"),
        )
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ConfigurationError("algorithm.hidden_sizes must contain positive integers.")
        object.__setattr__(
            self,
            "initial_log_standard_deviation",
            _number(
                self.initial_log_standard_deviation,
                path="algorithm.initial_log_standard_deviation",
            ),
        )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("algorithm.device must be one of: auto, cpu, cuda.")

    @property
    def num_updates(self) -> int:
        """Number of collect-update cycles."""
        return self.total_steps // self.rollout_steps


@dataclass(frozen=True, slots=True)
class ContinuousBanditConfig:
    """Definition and evaluation settings for the learnable reference task."""

    target_low: float
    target_high: float
    success_tolerance: float
    evaluation_episodes: int

    def __post_init__(self) -> None:
        low = _number(self.target_low, path="task.target_low")
        high = _number(self.target_high, path="task.target_high")
        if not -1.0 < low < high < 1.0:
            raise ConfigurationError(
                "task target bounds must satisfy -1 < target_low < target_high < 1."
            )
        tolerance = _positive_number(self.success_tolerance, path="task.success_tolerance")
        if tolerance >= high - low:
            raise ConfigurationError(
                "task.success_tolerance must be smaller than the target range."
            )
        object.__setattr__(self, "target_low", low)
        object.__setattr__(self, "target_high", high)
        object.__setattr__(self, "success_tolerance", tolerance)
        object.__setattr__(
            self,
            "evaluation_episodes",
            _integer(self.evaluation_episodes, path="task.evaluation_episodes"),
        )


@dataclass(frozen=True, slots=True)
class PPOExperimentConfig:
    """Complete single-agent PPO reference experiment."""

    schema_version: int
    name: str
    seed: int
    algorithm: PPOConfig
    task: ContinuousBanditConfig

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, path="schema_version")
        if version != 1:
            raise ConfigurationError(f"unsupported schema_version {version}; expected 1.")
        if not _EXPERIMENT_NAME.fullmatch(self.name):
            raise ConfigurationError("name must be a lowercase kebab-case identifier.")
        try:
            seed = validate_seed(self.seed)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "seed", seed)


def ppo_config_from_mapping(value: object) -> PPOConfig:
    """Build validated PPO hyperparameters from an untrusted algorithm mapping."""
    algorithm = _mapping(value, path="algorithm")
    _validate_keys(algorithm, path="algorithm", required=set(PPO_ALGORITHM_KEYS))
    return PPOConfig(
        total_steps=_integer(algorithm["total_steps"], path="algorithm.total_steps"),
        rollout_steps=_integer(algorithm["rollout_steps"], path="algorithm.rollout_steps"),
        update_epochs=_integer(algorithm["update_epochs"], path="algorithm.update_epochs"),
        minibatch_size=_integer(algorithm["minibatch_size"], path="algorithm.minibatch_size"),
        learning_rate=_number(algorithm["learning_rate"], path="algorithm.learning_rate"),
        gamma=_number(algorithm["gamma"], path="algorithm.gamma"),
        gae_lambda=_number(algorithm["gae_lambda"], path="algorithm.gae_lambda"),
        clip_coefficient=_number(
            algorithm["clip_coefficient"],
            path="algorithm.clip_coefficient",
        ),
        value_coefficient=_number(
            algorithm["value_coefficient"],
            path="algorithm.value_coefficient",
        ),
        entropy_coefficient=_number(
            algorithm["entropy_coefficient"],
            path="algorithm.entropy_coefficient",
        ),
        max_gradient_norm=_number(
            algorithm["max_gradient_norm"],
            path="algorithm.max_gradient_norm",
        ),
        hidden_sizes=_hidden_sizes(algorithm["hidden_sizes"]),
        initial_log_standard_deviation=_number(
            algorithm["initial_log_standard_deviation"],
            path="algorithm.initial_log_standard_deviation",
        ),
        device=_device_name(algorithm["device"]),
    )


def ppo_experiment_config_from_mapping(value: object) -> PPOExperimentConfig:
    """Build a validated PPO experiment from an untrusted mapping."""
    root = _mapping(value, path="configuration")
    _validate_keys(
        root,
        path="configuration",
        required={"schema_version", "name", "seed", "algorithm", "task"},
    )
    algorithm_config = ppo_config_from_mapping(root["algorithm"])

    task = _mapping(root["task"], path="task")
    _validate_keys(
        task,
        path="task",
        required={
            "target_low",
            "target_high",
            "success_tolerance",
            "evaluation_episodes",
        },
    )
    task_config = ContinuousBanditConfig(
        target_low=_number(task["target_low"], path="task.target_low"),
        target_high=_number(task["target_high"], path="task.target_high"),
        success_tolerance=_number(
            task["success_tolerance"],
            path="task.success_tolerance",
        ),
        evaluation_episodes=_integer(
            task["evaluation_episodes"],
            path="task.evaluation_episodes",
        ),
    )
    return PPOExperimentConfig(
        schema_version=_integer(root["schema_version"], path="schema_version"),
        name=_string(root["name"], path="name"),
        seed=_integer(root["seed"], path="seed", minimum=0),
        algorithm=algorithm_config,
        task=task_config,
    )


def load_ppo_experiment_config(path: str | Path) -> PPOExperimentConfig:
    """Safely load one self-contained PPO experiment YAML file."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("PPO configuration must use a .yaml or .yml extension.")
    try:
        contents = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"could not read configuration {config_path}: {error}") from error
    try:
        raw = cast(object, yaml.safe_load(contents))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {config_path}: {error}") from error
    return ppo_experiment_config_from_mapping(raw)
