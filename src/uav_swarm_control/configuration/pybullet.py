"""Validated configuration for the pinned PyBullet UAV simulator adapter."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.loading import experiment_config_from_mapping
from uav_swarm_control.configuration.models import (
    ConfigurationError,
    ExperimentConfig,
    experiment_config_to_dict,
)


class PyBulletDroneModel(StrEnum):
    """Crazyflie layouts supported by the upstream PID controller."""

    CF2X = "cf2x"
    CF2P = "cf2p"


class PyBulletPhysics(StrEnum):
    """Physics modes exposed by the pinned simulator."""

    BASE = "pyb"
    GROUND_EFFECT = "pyb_gnd"
    DRAG = "pyb_drag"
    DOWNWASH = "pyb_dw"
    ALL_EFFECTS = "pyb_gnd_drag_dw"


def _mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{path} keys must be strings.")
        result[key] = item
    return result


def _enum_value[T: StrEnum](value: object, enum_type: type[T], *, path: str) -> T:
    if not isinstance(value, str):
        raise ConfigurationError(f"{path} must be a string.")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ConfigurationError(f"{path} must be one of: {choices}.") from error


def _positive_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer.")
    if value < 1:
        raise ConfigurationError(f"{path} must be at least 1.")
    return value


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} must be a boolean.")
    return value


@dataclass(frozen=True, slots=True)
class PyBulletSimulatorConfig:
    """Simulator, vehicle, rendering, and two-rate timing choices."""

    drone_model: PyBulletDroneModel
    physics: PyBulletPhysics
    physics_frequency_hz: int
    control_frequency_hz: int
    gui: bool
    record_video: bool

    def __post_init__(self) -> None:
        model = _enum_value(self.drone_model, PyBulletDroneModel, path="simulator.drone_model")
        physics = _enum_value(self.physics, PyBulletPhysics, path="simulator.physics")
        physics_frequency = _positive_integer(
            self.physics_frequency_hz,
            path="simulator.physics_frequency_hz",
        )
        control_frequency = _positive_integer(
            self.control_frequency_hz,
            path="simulator.control_frequency_hz",
        )
        if physics_frequency % control_frequency != 0:
            raise ConfigurationError(
                "simulator.physics_frequency_hz must be divisible by control_frequency_hz."
            )
        object.__setattr__(self, "drone_model", model)
        object.__setattr__(self, "physics", physics)
        object.__setattr__(self, "physics_frequency_hz", physics_frequency)
        object.__setattr__(self, "control_frequency_hz", control_frequency)
        object.__setattr__(self, "gui", _boolean(self.gui, path="simulator.gui"))
        object.__setattr__(
            self,
            "record_video",
            _boolean(self.record_video, path="simulator.record_video"),
        )

    @property
    def physics_steps_per_control(self) -> int:
        """Number of rigid-body updates performed for one controller action."""
        return self.physics_frequency_hz // self.control_frequency_hz


@dataclass(frozen=True, slots=True)
class PyBulletExperimentConfig:
    """One common formation task executed by the PyBullet UAV adapter."""

    experiment: ExperimentConfig
    simulator: PyBulletSimulatorConfig

    def __post_init__(self) -> None:
        expected_step = 1.0 / self.simulator.control_frequency_hz
        if not math.isclose(
            self.experiment.environment.time_step_seconds,
            expected_step,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConfigurationError(
                "environment.time_step_seconds must equal 1 / simulator.control_frequency_hz."
            )


def pybullet_experiment_config_from_mapping(value: object) -> PyBulletExperimentConfig:
    """Build a strict simulator configuration while reusing the common task schema."""
    root = _mapping(value, path="configuration")
    if "simulator" not in root:
        raise ConfigurationError("configuration requires a simulator mapping.")
    simulator_values = _mapping(root["simulator"], path="simulator")
    expected_keys = {
        "drone_model",
        "physics",
        "physics_frequency_hz",
        "control_frequency_hz",
        "gui",
        "record_video",
    }
    missing = expected_keys - simulator_values.keys()
    unknown = simulator_values.keys() - expected_keys
    if missing:
        raise ConfigurationError(
            f"simulator is missing required keys: {', '.join(sorted(missing))}."
        )
    if unknown:
        raise ConfigurationError(f"simulator contains unknown keys: {', '.join(sorted(unknown))}.")

    experiment_values = dict(root)
    del experiment_values["simulator"]
    return PyBulletExperimentConfig(
        experiment=experiment_config_from_mapping(experiment_values),
        simulator=PyBulletSimulatorConfig(
            drone_model=_enum_value(
                simulator_values["drone_model"],
                PyBulletDroneModel,
                path="simulator.drone_model",
            ),
            physics=_enum_value(
                simulator_values["physics"],
                PyBulletPhysics,
                path="simulator.physics",
            ),
            physics_frequency_hz=_positive_integer(
                simulator_values["physics_frequency_hz"],
                path="simulator.physics_frequency_hz",
            ),
            control_frequency_hz=_positive_integer(
                simulator_values["control_frequency_hz"],
                path="simulator.control_frequency_hz",
            ),
            gui=_boolean(simulator_values["gui"], path="simulator.gui"),
            record_video=_boolean(
                simulator_values["record_video"],
                path="simulator.record_video",
            ),
        ),
    )


def pybullet_experiment_config_to_dict(config: PyBulletExperimentConfig) -> dict[str, object]:
    """Return the fully resolved task and simulator settings."""
    result = experiment_config_to_dict(config.experiment)
    result["simulator"] = {
        "drone_model": config.simulator.drone_model.value,
        "physics": config.simulator.physics.value,
        "physics_frequency_hz": config.simulator.physics_frequency_hz,
        "control_frequency_hz": config.simulator.control_frequency_hz,
        "physics_steps_per_control": config.simulator.physics_steps_per_control,
        "gui": config.simulator.gui,
        "record_video": config.simulator.record_video,
    }
    return result


def load_pybullet_experiment_config(path: str | Path) -> PyBulletExperimentConfig:
    """Safely load one self-contained PyBullet experiment YAML file."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("PyBullet configuration must use a .yaml or .yml extension.")
    try:
        contents = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"could not read configuration {config_path}: {error}") from error
    try:
        raw = cast(object, yaml.safe_load(contents))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {config_path}: {error}") from error
    return pybullet_experiment_config_from_mapping(raw)
