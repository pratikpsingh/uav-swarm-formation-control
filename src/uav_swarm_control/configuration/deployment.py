"""Strict policy-compression configuration."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.models import ConfigurationError
from uav_swarm_control.deployment.contracts import DeploymentArchitecture
from uav_swarm_control.seeding import validate_seed

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class TeacherConditionGate:
    """Minimum acceptable teacher behavior on one named neighbor study condition."""

    condition: str
    minimum_collision_free_success: float
    maximum_collision_any: float
    maximum_mean_normalized_shape_rmse: float

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.condition):
            raise ConfigurationError("teacher gate condition must use kebab-case.")
        if not 0.0 <= self.minimum_collision_free_success <= 1.0:
            raise ConfigurationError("teacher minimum success must be between zero and one.")
        if not 0.0 <= self.maximum_collision_any <= 1.0:
            raise ConfigurationError("teacher maximum collision rate must be between zero and one.")
        if self.maximum_mean_normalized_shape_rmse < 0.0 or not math.isfinite(
            self.maximum_mean_normalized_shape_rmse
        ):
            raise ConfigurationError(
                "teacher maximum formation error must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class CompressionCandidateConfig:
    """One controlled architecture/compression treatment."""

    name: str
    architecture: DeploymentArchitecture
    width: int
    depth: int
    structured_pruning_fraction: float
    quantization: str

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ConfigurationError("candidate name must use kebab-case.")
        if (
            isinstance(self.width, bool)
            or isinstance(self.depth, bool)
            or self.width < 1
            or self.depth < 1
        ):
            raise ConfigurationError("candidate width and depth must be positive integers.")
        if not 0.0 <= self.structured_pruning_fraction < 1.0:
            raise ConfigurationError("structured_pruning_fraction must be in [0, 1).")
        if self.quantization not in {"none", "int8-dynamic"}:
            raise ConfigurationError("candidate quantization must be none or int8-dynamic.")
        if self.architecture is not DeploymentArchitecture.FEED_FORWARD and (
            self.depth != 1 or self.structured_pruning_fraction or self.quantization != "none"
        ):
            raise ConfigurationError(
                "recurrent candidates use depth 1 without pruning or INT8 in this study."
            )


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Episode-level split and optimization settings for behavior cloning."""

    dataset_episodes_per_condition: int
    validation_fraction: float
    epochs: int
    fine_tune_epochs: int
    batch_size: int
    learning_rate: float

    def __post_init__(self) -> None:
        integers = (
            self.dataset_episodes_per_condition,
            self.epochs,
            self.fine_tune_epochs,
            self.batch_size,
        )
        if any(isinstance(value, bool) or value < 1 for value in integers):
            raise ConfigurationError("distillation counts must be positive integers.")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ConfigurationError("validation_fraction must be between zero and 0.5.")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ConfigurationError("distillation learning_rate must be finite and positive.")


@dataclass(frozen=True, slots=True)
class HostBenchmarkConfig:
    """Repeatable single-agent host latency protocol."""

    warmup_iterations: int
    measured_iterations: int
    torch_threads: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or value < 1
            for value in (self.warmup_iterations, self.measured_iterations, self.torch_threads)
        ):
            raise ConfigurationError("benchmark counts and torch_threads must be positive.")


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """External power-measurement status; software estimates are intentionally forbidden."""

    status: str
    reason: str | None
    joules_per_inference: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.status not in {"unavailable", "measured-externally"}:
            raise ConfigurationError("energy status must be unavailable or measured-externally.")
        if self.status == "unavailable" and not self.reason:
            raise ConfigurationError("unavailable energy measurement requires a reason.")
        if self.status == "measured-externally" and not self.joules_per_inference:
            raise ConfigurationError("external energy mode requires candidate measurements.")
        if any(
            value <= 0.0 or not math.isfinite(value) for value in self.joules_per_inference.values()
        ):
            raise ConfigurationError("energy measurements must be finite and positive.")


@dataclass(frozen=True, slots=True)
class DeploymentStudyConfig:
    """Validation gate, candidates, replications, and measurement protocol."""

    name: str
    teacher_gates: tuple[TeacherConditionGate, ...]
    candidates: tuple[CompressionCandidateConfig, ...]
    distillation_seeds: tuple[int, ...]
    dataset_seed: int
    evaluation_episodes_per_condition: int
    distillation: DistillationConfig
    benchmark: HostBenchmarkConfig
    energy: EnergyConfig
    profile: str

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ConfigurationError("deployment study name must use kebab-case.")
        if not self.teacher_gates or len({gate.condition for gate in self.teacher_gates}) != len(
            self.teacher_gates
        ):
            raise ConfigurationError("teacher gates must be non-empty and uniquely named.")
        names = [candidate.name for candidate in self.candidates]
        if not names or len(set(names)) != len(names):
            raise ConfigurationError("compression candidates must be non-empty and uniquely named.")
        if set(self.energy.joules_per_inference) - set(names):
            raise ConfigurationError("energy measurements reference unknown candidates.")
        plain_ff_widths = {
            candidate.width
            for candidate in self.candidates
            if candidate.architecture is DeploymentArchitecture.FEED_FORWARD
            and not candidate.structured_pruning_fraction
            and candidate.quantization == "none"
        }
        architectures = {candidate.architecture for candidate in self.candidates}
        if len(plain_ff_widths) < 3:
            raise ConfigurationError("study requires at least three plain feed-forward widths.")
        if architectures != set(DeploymentArchitecture):
            raise ConfigurationError("study must compare feed-forward, GRU, and LSTM actors.")
        if not any(candidate.structured_pruning_fraction for candidate in self.candidates):
            raise ConfigurationError("study requires a structured-pruning candidate.")
        if not any(candidate.quantization == "int8-dynamic" for candidate in self.candidates):
            raise ConfigurationError("study requires an INT8 candidate.")
        if self.profile not in {"smoke", "research"}:
            raise ConfigurationError("protocol.profile must be smoke or research.")
        if len(set(self.distillation_seeds)) != len(self.distillation_seeds):
            raise ConfigurationError("distillation seeds must be distinct.")
        if len(self.distillation_seeds) < 2:
            raise ConfigurationError("at least two distillation seeds are required.")
        if self.profile == "research" and len(self.distillation_seeds) < 5:
            raise ConfigurationError("research requires at least five distinct distillation seeds.")
        try:
            seeds = tuple(validate_seed(seed) for seed in self.distillation_seeds)
            dataset_seed = validate_seed(self.dataset_seed)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error
        if (
            isinstance(self.evaluation_episodes_per_condition, bool)
            or self.evaluation_episodes_per_condition < 1
        ):
            raise ConfigurationError("evaluation episodes per condition must be positive.")
        object.__setattr__(self, "distillation_seeds", seeds)
        object.__setattr__(self, "dataset_seed", dataset_seed)


def deployment_smoke_config(config: DeploymentStudyConfig) -> DeploymentStudyConfig:
    """Bound runtime without removing an architecture, compression method, or seed."""
    return replace(
        config,
        distillation=replace(
            config.distillation,
            dataset_episodes_per_condition=1,
            epochs=2,
            fine_tune_epochs=1,
        ),
        benchmark=replace(config.benchmark, warmup_iterations=5, measured_iterations=20),
        evaluation_episodes_per_condition=1,
        profile="smoke",
    )


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping.")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ConfigurationError(f"{path} keys must be strings.")
        result[key] = item
    return result


def _keys(values: Mapping[str, object], path: str, required: set[str]) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required
    if missing:
        raise ConfigurationError(f"{path} is missing required keys: {', '.join(sorted(missing))}.")
    if unknown:
        raise ConfigurationError(f"{path} contains unknown keys: {', '.join(sorted(unknown))}.")


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{path} must be a positive integer.")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{path} must be finite.")
    return result


def deployment_study_config_from_mapping(value: object) -> DeploymentStudyConfig:
    """Parse a closed schema so misspelled research settings cannot be ignored."""
    root = _mapping(value, "configuration")
    _keys(
        root,
        "configuration",
        {
            "schema_version",
            "name",
            "teacher_gate",
            "candidates",
            "distillation",
            "benchmark",
            "energy",
            "protocol",
        },
    )
    if root["schema_version"] != 1:
        raise ConfigurationError("deployment schema_version must be 1.")
    gate_values = root["teacher_gate"]
    if not isinstance(gate_values, list):
        raise ConfigurationError("teacher_gate must be a YAML list.")
    gates: list[TeacherConditionGate] = []
    for index, raw in enumerate(cast(list[object], gate_values)):
        path = f"teacher_gate[{index}]"
        item = _mapping(raw, path)
        _keys(
            item,
            path,
            {
                "condition",
                "minimum_collision_free_success",
                "maximum_collision_any",
                "maximum_mean_normalized_shape_rmse",
            },
        )
        gates.append(
            TeacherConditionGate(
                cast(str, item["condition"]),
                _number(
                    item["minimum_collision_free_success"], f"{path}.minimum_collision_free_success"
                ),
                _number(item["maximum_collision_any"], f"{path}.maximum_collision_any"),
                _number(
                    item["maximum_mean_normalized_shape_rmse"],
                    f"{path}.maximum_mean_normalized_shape_rmse",
                ),
            )
        )
    candidate_values = root["candidates"]
    if not isinstance(candidate_values, list):
        raise ConfigurationError("candidates must be a YAML list.")
    candidates: list[CompressionCandidateConfig] = []
    for index, raw in enumerate(cast(list[object], candidate_values)):
        path = f"candidates[{index}]"
        item = _mapping(raw, path)
        _keys(
            item,
            path,
            {
                "name",
                "architecture",
                "width",
                "depth",
                "structured_pruning_fraction",
                "quantization",
            },
        )
        try:
            architecture = DeploymentArchitecture(cast(str, item["architecture"]))
        except ValueError as error:
            raise ConfigurationError(f"{path}.architecture is invalid.") from error
        candidates.append(
            CompressionCandidateConfig(
                cast(str, item["name"]),
                architecture,
                _integer(item["width"], f"{path}.width"),
                _integer(item["depth"], f"{path}.depth"),
                _number(item["structured_pruning_fraction"], f"{path}.structured_pruning_fraction"),
                cast(str, item["quantization"]),
            )
        )
    distillation = _mapping(root["distillation"], "distillation")
    _keys(
        distillation,
        "distillation",
        {
            "dataset_episodes_per_condition",
            "validation_fraction",
            "epochs",
            "fine_tune_epochs",
            "batch_size",
            "learning_rate",
        },
    )
    benchmark = _mapping(root["benchmark"], "benchmark")
    _keys(benchmark, "benchmark", {"warmup_iterations", "measured_iterations", "torch_threads"})
    energy = _mapping(root["energy"], "energy")
    _keys(energy, "energy", {"status", "reason", "joules_per_inference"})
    raw_energy = _mapping(energy["joules_per_inference"], "energy.joules_per_inference")
    protocol = _mapping(root["protocol"], "protocol")
    _keys(
        protocol,
        "protocol",
        {"distillation_seeds", "dataset_seed", "evaluation_episodes_per_condition", "profile"},
    )
    raw_seeds = protocol["distillation_seeds"]
    if not isinstance(raw_seeds, list):
        raise ConfigurationError("protocol.distillation_seeds must be a YAML list.")
    return DeploymentStudyConfig(
        cast(str, root["name"]),
        tuple(gates),
        tuple(candidates),
        tuple(cast(list[int], raw_seeds)),
        cast(int, protocol["dataset_seed"]),
        _integer(
            protocol["evaluation_episodes_per_condition"],
            "protocol.evaluation_episodes_per_condition",
        ),
        DistillationConfig(
            _integer(
                distillation["dataset_episodes_per_condition"],
                "distillation.dataset_episodes_per_condition",
            ),
            _number(distillation["validation_fraction"], "distillation.validation_fraction"),
            _integer(distillation["epochs"], "distillation.epochs"),
            _integer(distillation["fine_tune_epochs"], "distillation.fine_tune_epochs"),
            _integer(distillation["batch_size"], "distillation.batch_size"),
            _number(distillation["learning_rate"], "distillation.learning_rate"),
        ),
        HostBenchmarkConfig(
            _integer(benchmark["warmup_iterations"], "benchmark.warmup_iterations"),
            _integer(benchmark["measured_iterations"], "benchmark.measured_iterations"),
            _integer(benchmark["torch_threads"], "benchmark.torch_threads"),
        ),
        EnergyConfig(
            cast(str, energy["status"]),
            cast(str | None, energy["reason"]),
            {
                key: _number(item, f"energy.joules_per_inference.{key}")
                for key, item in raw_energy.items()
            },
        ),
        cast(str, protocol["profile"]),
    )


def load_deployment_study_config(path: str | Path) -> DeploymentStudyConfig:
    """Safely load one policy-compression study configuration."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("deployment configuration must use .yaml or .yml.")
    try:
        raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot load deployment configuration: {error}") from error
    return deployment_study_config_from_mapping(raw)


__all__ = [
    "CompressionCandidateConfig",
    "DeploymentStudyConfig",
    "DistillationConfig",
    "EnergyConfig",
    "HostBenchmarkConfig",
    "TeacherConditionGate",
    "deployment_smoke_config",
    "deployment_study_config_from_mapping",
    "load_deployment_study_config",
]
