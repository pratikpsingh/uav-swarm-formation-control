"""Strict configuration for the clean-room native-DMPC trajectory-planner adaptation."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.models import ConfigurationError

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


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


def _integer(value: object, *, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{path} must be an integer at least {minimum}.")
    return value


def _positive(value: object, *, path: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigurationError(f"{path} must be finite and {qualifier}.")
    return result


@dataclass(frozen=True, slots=True)
class DMPCWeights:
    """Quadratic costs applied to the finite-horizon trajectory."""

    terminal_position: float
    velocity: float
    acceleration: float
    jerk: float

    def __post_init__(self) -> None:
        for name in ("terminal_position", "velocity", "acceleration", "jerk"):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), path=f"weights.{name}", allow_zero=True),
            )
        if self.terminal_position == 0.0:
            raise ConfigurationError("weights.terminal_position must be positive.")


@dataclass(frozen=True, slots=True)
class DMPCLimits:
    """Planner limits; velocity is inherited from the shared task configuration."""

    acceleration_mps2: float
    jerk_mps3: float
    minimum_separation_m: float
    downwash_scaling: float

    def __post_init__(self) -> None:
        for name in (
            "acceleration_mps2",
            "jerk_mps3",
            "minimum_separation_m",
            "downwash_scaling",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), path=f"limits.{name}"))


@dataclass(frozen=True, slots=True)
class DMPCSolverConfig:
    """Numerical convergence controls for SciPy SLSQP."""

    max_iterations: int
    function_tolerance: float
    constraint_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_iterations", _integer(self.max_iterations, path="solver.max_iterations")
        )
        for name in ("function_tolerance", "constraint_tolerance"):
            object.__setattr__(self, name, _positive(getattr(self, name), path=f"solver.{name}"))


@dataclass(frozen=True, slots=True)
class DMPCConfig:
    """Controller-only settings composed with a Stage 7 task at runtime."""

    schema_version: int
    name: str
    prediction_horizon_steps: int
    planning_interval_seconds: float
    weights: DMPCWeights
    limits: DMPCLimits
    solver: DMPCSolverConfig

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError("schema_version must equal 1.")
        if _NAME.fullmatch(self.name) is None:
            raise ConfigurationError("name must use lowercase kebab-case.")
        object.__setattr__(
            self,
            "prediction_horizon_steps",
            _integer(self.prediction_horizon_steps, path="prediction_horizon_steps", minimum=2),
        )
        object.__setattr__(
            self,
            "planning_interval_seconds",
            _positive(self.planning_interval_seconds, path="planning_interval_seconds"),
        )


def dmpc_config_from_mapping(value: object) -> DMPCConfig:
    """Build a DMPC configuration while rejecting silent defaults and unknown keys."""
    root = _mapping(value, path="configuration")
    required = {
        "schema_version",
        "name",
        "prediction_horizon_steps",
        "planning_interval_seconds",
        "weights",
        "limits",
        "solver",
    }
    _keys(root, path="configuration", required=required)
    weights = _mapping(root["weights"], path="weights")
    _keys(
        weights,
        path="weights",
        required={"terminal_position", "velocity", "acceleration", "jerk"},
    )
    limits = _mapping(root["limits"], path="limits")
    _keys(
        limits,
        path="limits",
        required={
            "acceleration_mps2",
            "jerk_mps3",
            "minimum_separation_m",
            "downwash_scaling",
        },
    )
    solver = _mapping(root["solver"], path="solver")
    _keys(
        solver,
        path="solver",
        required={"max_iterations", "function_tolerance", "constraint_tolerance"},
    )
    return DMPCConfig(
        schema_version=_integer(root["schema_version"], path="schema_version"),
        name=cast(str, root["name"]),
        prediction_horizon_steps=_integer(
            root["prediction_horizon_steps"], path="prediction_horizon_steps", minimum=2
        ),
        planning_interval_seconds=_positive(
            root["planning_interval_seconds"], path="planning_interval_seconds"
        ),
        weights=DMPCWeights(
            terminal_position=_positive(
                weights["terminal_position"], path="weights.terminal_position", allow_zero=True
            ),
            velocity=_positive(weights["velocity"], path="weights.velocity", allow_zero=True),
            acceleration=_positive(
                weights["acceleration"], path="weights.acceleration", allow_zero=True
            ),
            jerk=_positive(weights["jerk"], path="weights.jerk", allow_zero=True),
        ),
        limits=DMPCLimits(
            acceleration_mps2=_positive(
                limits["acceleration_mps2"], path="limits.acceleration_mps2"
            ),
            jerk_mps3=_positive(limits["jerk_mps3"], path="limits.jerk_mps3"),
            minimum_separation_m=_positive(
                limits["minimum_separation_m"], path="limits.minimum_separation_m"
            ),
            downwash_scaling=_positive(limits["downwash_scaling"], path="limits.downwash_scaling"),
        ),
        solver=DMPCSolverConfig(
            max_iterations=_integer(solver["max_iterations"], path="solver.max_iterations"),
            function_tolerance=_positive(
                solver["function_tolerance"], path="solver.function_tolerance"
            ),
            constraint_tolerance=_positive(
                solver["constraint_tolerance"], path="solver.constraint_tolerance"
            ),
        ),
    )


def load_dmpc_config(path: str | Path) -> DMPCConfig:
    """Load a self-contained DMPC controller YAML file safely."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("DMPC configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load DMPC configuration: {error}") from error
    return dmpc_config_from_mapping(raw)
