"""Strict evidence configuration for cross-paper reports."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

from uav_swarm_control.configuration.models import ConfigurationError

_KEBAB_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class EvidenceStatus(StrEnum):
    """How an evidence value was obtained."""

    REPORTED = "reported"
    MEASURED = "measured"
    NOT_REPORTED = "not-reported"
    NOT_APPLICABLE = "not-applicable"


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    """A value whose absence and provenance are both explicit."""

    status: EvidenceStatus
    value: str
    source: str

    def __post_init__(self) -> None:
        try:
            status = EvidenceStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ConfigurationError("evidence status is invalid.") from error
        if not self.value.strip():
            raise ConfigurationError("evidence value must not be empty.")
        source_id, separator, locator = self.source.partition(":")
        if not separator or not source_id or not locator.strip():
            raise ConfigurationError("evidence source must use 'source-id: locator'.")
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """One immutable local file used by the evidence catalog."""

    identifier: str
    title: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if _KEBAB_NAME.fullmatch(self.identifier) is None:
            raise ConfigurationError("source id must be an alphanumeric kebab-case identifier.")
        if not self.title.strip() or self.path.is_absolute() or ".." in self.path.parts:
            raise ConfigurationError(
                "source title must be set and path must stay under evidence-root."
            )
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ConfigurationError(
                "source sha256 must contain 64 lowercase hexadecimal characters."
            )


@dataclass(frozen=True, slots=True)
class NativeSystemEvidence:
    """Paper-native or repository-native system context; never a controlled result."""

    identifier: str
    label: str
    method_family: str
    environment: EvidenceValue
    simulator_version: EvidenceValue
    action_semantics: EvidenceValue
    observation_access: EvidenceValue
    training_budget: EvidenceValue
    seeds: EvidenceValue
    uncertainty: EvidenceValue
    execution: EvidenceValue
    evaluation_protocol: EvidenceValue
    native_result: EvidenceValue
    comparability_note: str

    def __post_init__(self) -> None:
        if (
            _KEBAB_NAME.fullmatch(self.identifier) is None
            or not self.label.strip()
            or not self.method_family.strip()
        ):
            raise ConfigurationError("native system id, label, and method_family are required.")
        if not self.comparability_note.strip():
            raise ConfigurationError("native system comparability_note is required.")


@dataclass(frozen=True, slots=True)
class ControlledMethod:
    """Method-specific semantics added to compatible common-environment artifacts."""

    identifier: str
    label: str
    action_semantics: str
    observation_access: str
    execution: str

    def __post_init__(self) -> None:
        if _KEBAB_NAME.fullmatch(self.identifier) is None or any(
            not value.strip()
            for value in (
                self.identifier,
                self.label,
                self.action_semantics,
                self.observation_access,
                self.execution,
            )
        ):
            raise ConfigurationError("controlled method fields must not be empty.")


@dataclass(frozen=True, slots=True)
class CrossPaperComparisonConfig:
    """Complete cross-paper comparison evidence and controlled-result reporting contract."""

    name: str
    sources: tuple[EvidenceSource, ...]
    native_systems: tuple[NativeSystemEvidence, ...]
    controlled_methods: tuple[ControlledMethod, ...]
    expected_tasks: tuple[str, ...]
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        if _KEBAB_NAME.fullmatch(self.name) is None:
            raise ConfigurationError("comparison name must use lowercase kebab-case.")
        for values, label in (
            (self.sources, "sources"),
            (self.native_systems, "native_systems"),
            (self.controlled_methods, "controlled_methods"),
            (self.expected_tasks, "expected_tasks"),
            (self.metrics, "metrics"),
        ):
            if not values:
                raise ConfigurationError(f"comparison {label} must not be empty.")
        source_ids = [source.identifier for source in self.sources]
        native_ids = [system.identifier for system in self.native_systems]
        method_ids = [method.identifier for method in self.controlled_methods]
        for values, label in (
            (source_ids, "source"),
            (native_ids, "native system"),
            (method_ids, "controlled method"),
            (list(self.expected_tasks), "expected task"),
            (list(self.metrics), "metric"),
        ):
            if len(set(values)) != len(values):
                raise ConfigurationError(f"{label} identifiers must be unique.")
        if any(_KEBAB_NAME.fullmatch(task) is None for task in self.expected_tasks):
            raise ConfigurationError("expected task names must use lowercase kebab-case.")
        if any(not metric.isidentifier() for metric in self.metrics):
            raise ConfigurationError("metric names must be valid identifiers.")
        known_sources = set(source_ids)
        for system in self.native_systems:
            for field_name in _EVIDENCE_FIELDS:
                evidence = cast(EvidenceValue, getattr(system, field_name))
                if evidence.source.partition(":")[0] not in known_sources:
                    raise ConfigurationError(
                        f"native system {system.identifier!r} references an unknown source."
                    )


_EVIDENCE_FIELDS = (
    "environment",
    "simulator_version",
    "action_semantics",
    "observation_access",
    "training_budget",
    "seeds",
    "uncertainty",
    "execution",
    "evaluation_protocol",
    "native_result",
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


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path} must be a non-empty string.")
    return value


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a YAML list.")
    items = cast(list[object], value)
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(items))


def _evidence(value: object, path: str) -> EvidenceValue:
    item = _mapping(value, path)
    _keys(item, path, {"status", "value", "source"})
    return EvidenceValue(
        cast(EvidenceStatus, _string(item["status"], f"{path}.status")),
        _string(item["value"], f"{path}.value"),
        _string(item["source"], f"{path}.source"),
    )


def cross_paper_config_from_mapping(value: object) -> CrossPaperComparisonConfig:
    """Parse the closed cross-paper comparison schema."""
    root = _mapping(value, "configuration")
    _keys(
        root,
        "configuration",
        {
            "schema_version",
            "name",
            "sources",
            "native_systems",
            "controlled_methods",
            "expected_tasks",
            "metrics",
        },
    )
    if root["schema_version"] != 1:
        raise ConfigurationError("cross-paper schema_version must be 1.")

    raw_sources = root["sources"]
    if not isinstance(raw_sources, list):
        raise ConfigurationError("sources must be a YAML list.")
    sources: list[EvidenceSource] = []
    for index, raw in enumerate(cast(list[object], raw_sources)):
        path = f"sources[{index}]"
        item = _mapping(raw, path)
        _keys(item, path, {"id", "title", "path", "sha256"})
        sources.append(
            EvidenceSource(
                _string(item["id"], f"{path}.id"),
                _string(item["title"], f"{path}.title"),
                Path(_string(item["path"], f"{path}.path")),
                _string(item["sha256"], f"{path}.sha256"),
            )
        )

    raw_systems = root["native_systems"]
    if not isinstance(raw_systems, list):
        raise ConfigurationError("native_systems must be a YAML list.")
    systems: list[NativeSystemEvidence] = []
    for index, raw in enumerate(cast(list[object], raw_systems)):
        path = f"native_systems[{index}]"
        item = _mapping(raw, path)
        required = {"id", "label", "method_family", "comparability_note", *_EVIDENCE_FIELDS}
        _keys(item, path, required)
        systems.append(
            NativeSystemEvidence(
                identifier=_string(item["id"], f"{path}.id"),
                label=_string(item["label"], f"{path}.label"),
                method_family=_string(item["method_family"], f"{path}.method_family"),
                environment=_evidence(item["environment"], f"{path}.environment"),
                simulator_version=_evidence(item["simulator_version"], f"{path}.simulator_version"),
                action_semantics=_evidence(item["action_semantics"], f"{path}.action_semantics"),
                observation_access=_evidence(
                    item["observation_access"], f"{path}.observation_access"
                ),
                training_budget=_evidence(item["training_budget"], f"{path}.training_budget"),
                seeds=_evidence(item["seeds"], f"{path}.seeds"),
                uncertainty=_evidence(item["uncertainty"], f"{path}.uncertainty"),
                execution=_evidence(item["execution"], f"{path}.execution"),
                evaluation_protocol=_evidence(
                    item["evaluation_protocol"], f"{path}.evaluation_protocol"
                ),
                native_result=_evidence(item["native_result"], f"{path}.native_result"),
                comparability_note=_string(
                    item["comparability_note"], f"{path}.comparability_note"
                ),
            )
        )

    raw_methods = root["controlled_methods"]
    if not isinstance(raw_methods, list):
        raise ConfigurationError("controlled_methods must be a YAML list.")
    methods: list[ControlledMethod] = []
    for index, raw in enumerate(cast(list[object], raw_methods)):
        path = f"controlled_methods[{index}]"
        item = _mapping(raw, path)
        _keys(item, path, {"id", "label", "action_semantics", "observation_access", "execution"})
        methods.append(
            ControlledMethod(
                _string(item["id"], f"{path}.id"),
                _string(item["label"], f"{path}.label"),
                _string(item["action_semantics"], f"{path}.action_semantics"),
                _string(item["observation_access"], f"{path}.observation_access"),
                _string(item["execution"], f"{path}.execution"),
            )
        )

    return CrossPaperComparisonConfig(
        name=_string(root["name"], "name"),
        sources=tuple(sources),
        native_systems=tuple(systems),
        controlled_methods=tuple(methods),
        expected_tasks=_string_tuple(root["expected_tasks"], "expected_tasks"),
        metrics=_string_tuple(root["metrics"], "metrics"),
    )


def load_cross_paper_config(path: str | Path) -> CrossPaperComparisonConfig:
    """Load a YAML evidence catalog with source-friendly errors."""
    config_path = Path(path)
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError("cross-paper configuration must use a .yaml or .yml extension.")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"could not load {config_path}: {error}") from error
    return cross_paper_config_from_mapping(value)
