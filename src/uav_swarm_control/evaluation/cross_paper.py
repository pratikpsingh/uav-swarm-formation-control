"""Auditable two-track report generation for Stage 13."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from uav_swarm_control.configuration.cross_paper import (
    CrossPaperComparisonConfig,
    EvidenceValue,
)
from uav_swarm_control.evaluation.provenance import collect_provenance, write_source_snapshot


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return {str(key): item for key, item in cast(Mapping[object, object], value).items()}


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number.")
    return float(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evidence_sources(
    config: CrossPaperComparisonConfig, evidence_root: Path
) -> list[dict[str, object]]:
    """Verify every local source before trusting any paper-native metadata."""
    root = evidence_root.resolve()
    records: list[dict[str, object]] = []
    for source in config.sources:
        path = (root / source.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"evidence source escapes evidence root: {source.path}")
        if not path.is_file():
            raise FileNotFoundError(f"evidence source does not exist: {path}")
        observed = _sha256(path)
        if observed != source.sha256:
            raise ValueError(
                f"evidence source checksum changed for {source.identifier}: "
                f"expected {source.sha256}, observed {observed}."
            )
        records.append(
            {
                "id": source.identifier,
                "title": source.title,
                "path": str(source.path),
                "sha256": observed,
                "verified": True,
            }
        )
    return records


def _evidence_record(value: EvidenceValue) -> dict[str, str]:
    return {"status": value.status.value, "value": value.value, "source": value.source}


def native_evidence_rows(config: CrossPaperComparisonConfig) -> list[dict[str, object]]:
    """Normalize the native evidence track without asserting cross-system comparability."""
    rows: list[dict[str, object]] = []
    for system in config.native_systems:
        rows.append(
            {
                "track": "native-system-context",
                "id": system.identifier,
                "method": system.label,
                "method_family": system.method_family,
                "environment": _evidence_record(system.environment),
                "simulator_version": _evidence_record(system.simulator_version),
                "action_semantics": _evidence_record(system.action_semantics),
                "observation_access": _evidence_record(system.observation_access),
                "training_budget": _evidence_record(system.training_budget),
                "seeds": _evidence_record(system.seeds),
                "uncertainty": _evidence_record(system.uncertainty),
                "execution": _evidence_record(system.execution),
                "evaluation_protocol": _evidence_record(system.evaluation_protocol),
                "native_result": _evidence_record(system.native_result),
                "comparability_note": system.comparability_note,
            }
        )
    return rows


def _resolve_reference(reference: str, owner: Path, project_root: Path) -> Path:
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [project_root / path, owner.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"artifact reference does not exist: {reference}")


def _metric_rows(
    comparison: Mapping[str, object], metric_names: Sequence[str]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    values = _mapping(comparison.get("metrics"), "comparison.metrics")
    mappo: dict[str, dict[str, float]] = {}
    dmpc: dict[str, dict[str, float]] = {}
    for name in metric_names:
        metric = _mapping(values.get(name), f"comparison.metrics.{name}")
        mappo[name] = {
            "mean": _number(metric.get("mappo_mean"), f"{name}.mappo_mean"),
            "sample_std": _number(
                metric.get("mappo_sample_std_across_training_seeds"),
                f"{name}.mappo_sample_std_across_training_seeds",
            ),
        }
        dmpc[name] = {
            "mean": _number(metric.get("dmpc_mean"), f"{name}.dmpc_mean"),
            "sample_std": _number(
                metric.get("dmpc_sample_std_across_evaluation_episodes"),
                f"{name}.dmpc_sample_std_across_evaluation_episodes",
            ),
        }
    return mappo, dmpc


def _environment_label(protocol: Mapping[str, object]) -> str:
    physics = _mapping(protocol.get("physics"), "comparison_protocol.physics")
    experiment = _mapping(physics.get("experiment"), "comparison_protocol.physics.experiment")
    formation = _mapping(experiment.get("formation"), "experiment.formation")
    environment = _mapping(experiment.get("environment"), "experiment.environment")
    simulator = _mapping(physics.get("simulator"), "comparison_protocol.physics.simulator")
    kind = _string(formation.get("kind"), "experiment.formation.kind")
    agents = _integer(formation.get("num_agents"), "experiment.formation.num_agents")
    physics_mode = _string(simulator.get("physics"), "simulator.physics")
    control_hz = _integer(simulator.get("control_frequency_hz"), "simulator.control_frequency_hz")
    horizon = _integer(environment.get("max_episode_steps"), "environment.max_episode_steps")
    return (
        f"gym-pybullet-drones; {kind}; {agents} UAV; CF2X/{physics_mode}; "
        f"{control_hz} Hz control; {horizon} steps"
    )


def _simulator_label(result: Mapping[str, object]) -> str:
    simulator = _mapping(result.get("simulator"), "dmpc_result.simulator")
    package = _string(simulator.get("simulator_package"), "simulator.simulator_package")
    version = _string(simulator.get("simulator_version"), "simulator.simulator_version")
    revision = _string(simulator.get("simulator_revision"), "simulator.simulator_revision")
    api_version = _integer(simulator.get("pybullet_api_version"), "simulator.pybullet_api_version")
    return f"{package} {version} @ {revision}; PyBullet API {api_version}"


def normalize_controlled_comparisons(
    config: CrossPaperComparisonConfig,
    comparison_paths: Sequence[Path],
    *,
    project_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Normalize guarded MAPPO/DMPC artifacts while retaining sampling-unit differences."""
    method_by_id = {method.identifier: method for method in config.controlled_methods}
    rows_by_task: dict[str, list[dict[str, object]]] = {}
    profiles: set[str] = set()
    fingerprints: dict[str, str] = {}

    for comparison_path in comparison_paths:
        path = comparison_path.resolve()
        comparison = _load_json(path, "controller comparison")
        if comparison.get("artifact_schema_version") != 1:
            raise ValueError(f"unsupported controller comparison schema in {path}.")
        fingerprint = _string(
            comparison.get("comparison_fingerprint"), "comparison.comparison_fingerprint"
        )
        mappo_path = _resolve_reference(
            _string(comparison.get("mappo_result"), "comparison.mappo_result"),
            path,
            project_root,
        )
        dmpc_path = _resolve_reference(
            _string(comparison.get("dmpc_result"), "comparison.dmpc_result"),
            path,
            project_root,
        )
        manifest_path = mappo_path.parent / "manifest.json"
        mappo = _load_json(mappo_path, "MAPPO summary")
        dmpc = _load_json(dmpc_path, "DMPC result")
        manifest = _load_json(manifest_path, "MAPPO manifest")
        protocol = _mapping(manifest.get("comparison_protocol"), "manifest.comparison_protocol")
        protocol_fingerprint = _string(protocol.get("fingerprint"), "protocol.fingerprint")
        if any(
            candidate != fingerprint
            for candidate in (
                mappo.get("comparison_fingerprint"),
                dmpc.get("comparison_fingerprint"),
                protocol_fingerprint,
            )
        ):
            raise ValueError(f"comparison fingerprint mismatch in {path}.")

        manifest_config = _mapping(manifest.get("configuration"), "manifest.configuration")
        mappo_config = _mapping(manifest_config.get("mappo"), "manifest.configuration.mappo")
        experiment = _mapping(
            mappo_config.get("experiment"), "manifest.configuration.mappo.experiment"
        )
        task = _string(experiment.get("name"), "experiment.name")
        if task not in config.expected_tasks:
            raise ValueError(f"unexpected controlled task {task!r}.")
        if task in rows_by_task:
            raise ValueError(f"duplicate controlled comparison for task {task!r}.")
        profile = _string(mappo.get("profile"), "mappo.profile")
        if profile not in {"smoke", "research"}:
            raise ValueError(f"unsupported controlled profile {profile!r}.")
        if profile != dmpc.get("profile"):
            raise ValueError(f"MAPPO and DMPC profile mismatch for task {task!r}.")
        profiles.add(profile)
        fingerprints[task] = fingerprint

        mappo_method_id = _string(manifest.get("method"), "manifest.method")
        dmpc_method_id = _string(dmpc.get("method"), "dmpc.method")
        if mappo_method_id not in method_by_id or dmpc_method_id not in method_by_id:
            raise ValueError(
                "controlled artifact contains a method absent from the Stage 13 config."
            )
        mappo_metrics, dmpc_metrics = _metric_rows(comparison, config.metrics)
        environment = _environment_label(protocol)
        simulator = _simulator_label(dmpc)
        seed_count = _integer(mappo.get("seed_count"), "mappo.seed_count")
        evaluation_episodes = _integer(
            mappo.get("evaluation_episodes_per_seed"), "mappo.evaluation_episodes_per_seed"
        )
        evaluation_seed = _integer(mappo.get("evaluation_root_seed"), "mappo.evaluation_root_seed")
        training_metrics = _mapping(mappo.get("metrics"), "mappo.metrics")
        training_steps = _mapping(
            training_metrics.get("training_environment_steps"),
            "mappo.metrics.training_environment_steps",
        )
        steps_per_seed = _number(training_steps.get("mean"), "training_environment_steps.mean")
        dmpc_summary = _mapping(dmpc.get("common_summary"), "dmpc.common_summary")
        dmpc_episodes = _integer(dmpc_summary.get("episode_count"), "dmpc.episode_count")

        mappo_method = method_by_id[mappo_method_id]
        dmpc_method = method_by_id[dmpc_method_id]
        common = {
            "track": "controlled-common-environment",
            "task": task,
            "profile": profile,
            "environment": environment,
            "simulator_version": simulator,
            "comparison_fingerprint": fingerprint,
        }
        rows_by_task[task] = [
            {
                **common,
                "method": mappo_method.label,
                "method_id": mappo_method.identifier,
                "action_semantics": mappo_method.action_semantics,
                "observation_access": mappo_method.observation_access,
                "training_budget": f"{steps_per_seed:g} environment transitions per seed",
                "seeds": (
                    f"{seed_count} training seeds; evaluation root {evaluation_seed}; "
                    f"{evaluation_episodes} episodes per trained policy"
                ),
                "uncertainty": (f"sample SD across {seed_count} independently trained seed means"),
                "execution": mappo_method.execution,
                "metrics": mappo_metrics,
                "source_artifact": str(path),
            },
            {
                **common,
                "method": dmpc_method.label,
                "method_id": dmpc_method.identifier,
                "action_semantics": dmpc_method.action_semantics,
                "observation_access": dmpc_method.observation_access,
                "training_budget": "not applicable - deterministic optimizer; no policy training",
                "seeds": (
                    f"no training seeds; evaluation root {evaluation_seed}; "
                    f"{dmpc_episodes} initial-condition episodes"
                ),
                "uncertainty": f"sample SD across {dmpc_episodes} evaluation episodes",
                "execution": dmpc_method.execution,
                "metrics": dmpc_metrics,
                "source_artifact": str(path),
            },
        ]

    missing_tasks = [task for task in config.expected_tasks if task not in rows_by_task]
    rows = [row for task in config.expected_tasks for row in rows_by_task.get(task, [])]
    coverage: dict[str, object] = {
        "expected_tasks": list(config.expected_tasks),
        "included_tasks": [task for task in config.expected_tasks if task in rows_by_task],
        "missing_tasks": missing_tasks,
        "profiles": sorted(profiles),
        "fingerprints_by_task": fingerprints,
        "research_ready": not missing_tasks and profiles == {"research"},
    }
    return rows, coverage


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _evidence_cell(value: object) -> str:
    evidence = _mapping(value, "native evidence")
    return _escape(f"[{evidence.get('status')}] {evidence.get('value')} ({evidence.get('source')})")


def _metric_cell(row: Mapping[str, object], name: str) -> str:
    metrics = _mapping(row.get("metrics"), "controlled row metrics")
    metric = _mapping(metrics.get(name), f"controlled row metrics.{name}")
    mean = _number(metric.get("mean"), name)
    sample_std = _number(metric.get("sample_std"), name)
    return f"{mean:.6g} (SD {sample_std:.3g})"


def render_markdown_report(
    config: CrossPaperComparisonConfig,
    native_rows: Sequence[Mapping[str, object]],
    controlled_rows: Sequence[Mapping[str, object]],
    coverage: Mapping[str, object],
    verified_sources: Sequence[Mapping[str, object]],
) -> str:
    """Render tables that always carry the Stage 13 context gate columns."""
    lines = [
        f"# {config.name}",
        "",
        "This report keeps controlled common-environment results separate from native-system "
        "context. Native values are not a leaderboard because the control interfaces, information "
        "sets, simulators, budgets, and metrics differ.",
        "",
        "## Controlled common-environment track",
        "",
    ]
    if controlled_rows:
        headers = [
            "Track",
            "Task/profile",
            "Method",
            "Environment",
            "Simulator version",
            "Action semantics",
            "Observation access",
            "Training budget",
            "Seeds",
            "Uncertainty",
            "Execution",
            *config.metrics,
        ]
        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        for row in controlled_rows:
            cells = [
                row["track"],
                f"{row['task']} / {row['profile']}",
                row["method"],
                row["environment"],
                row["simulator_version"],
                row["action_semantics"],
                row["observation_access"],
                row["training_budget"],
                row["seeds"],
                row["uncertainty"],
                row["execution"],
                *[_metric_cell(row, name) for name in config.metrics],
            ]
            lines.append("| " + " | ".join(_escape(cell) for cell in cells) + " |")
    else:
        lines.append("No compatible controlled comparison artifacts were supplied.")
    lines.extend(
        [
            "",
            "MAPPO and DMPC standard deviations have different sampling units; no significance "
            "claim is made from their numerical proximity. A smoke profile validates plumbing "
            "only.",
            "",
            "## Native-system context track",
            "",
        ]
    )
    native_headers = [
        "Track",
        "System",
        "Environment",
        "Simulator version",
        "Action semantics",
        "Observation access",
        "Training budget",
        "Seeds",
        "Uncertainty",
        "Execution",
        "Evaluation protocol",
        "Selected native result",
        "Comparability boundary",
    ]
    lines.extend(
        [
            "| " + " | ".join(native_headers) + " |",
            "| " + " | ".join("---" for _ in native_headers) + " |",
        ]
    )
    for row in native_rows:
        cells = [
            row["track"],
            f"{row['method']} ({row['method_family']})",
            _evidence_cell(row["environment"]),
            _evidence_cell(row["simulator_version"]),
            _evidence_cell(row["action_semantics"]),
            _evidence_cell(row["observation_access"]),
            _evidence_cell(row["training_budget"]),
            _evidence_cell(row["seeds"]),
            _evidence_cell(row["uncertainty"]),
            _evidence_cell(row["execution"]),
            _evidence_cell(row["evaluation_protocol"]),
            _evidence_cell(row["native_result"]),
            row["comparability_note"],
        ]
        lines.append("| " + " | ".join(_escape(cell) for cell in cells) + " |")
    lines.extend(
        [
            "",
            "## Evidence and readiness",
            "",
            f"- Research-ready controlled coverage: `{bool(coverage['research_ready'])}`.",
            f"- Missing controlled tasks: `{coverage['missing_tasks']}`.",
            f"- Included profiles: `{coverage['profiles']}`.",
        ]
    )
    for source in verified_sources:
        lines.append(
            f"- Verified `{source['id']}` at `{source['path']}` with SHA-256 `{source['sha256']}`."
        )
    return "\n".join(lines) + "\n"


def build_cross_paper_report(
    config: CrossPaperComparisonConfig,
    config_path: Path,
    comparison_paths: Sequence[Path],
    output_root: Path,
    *,
    project_root: Path,
    evidence_root: Path,
) -> Path:
    """Verify inputs and atomically write the Stage 13 report bundle."""
    verified = verify_evidence_sources(config, evidence_root)
    native_rows = native_evidence_rows(config)
    controlled_rows, coverage = normalize_controlled_comparisons(
        config, comparison_paths, project_root=project_root
    )
    report = {
        "artifact_schema_version": 1,
        "name": config.name,
        "research_ready": coverage["research_ready"],
        "coverage": coverage,
        "controlled_track": controlled_rows,
        "native_track": native_rows,
        "source_verification": verified,
        "warnings": [
            "Native-system values are contextual and are not directly comparable.",
            "MAPPO uncertainty is across trained seeds; DMPC uncertainty is across episodes.",
            "Smoke results are software checks, not scientific evidence.",
        ],
    }
    markdown = render_markdown_report(config, native_rows, controlled_rows, coverage, verified)
    config_path = config_path.resolve()
    input_hashes = {str(path.resolve()): _sha256(path.resolve()) for path in comparison_paths}
    manifest = {
        "artifact_schema_version": 1,
        "report_name": config.name,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "controlled_input_sha256": input_hashes,
        "evidence_sources": verified,
        "project_provenance": collect_provenance(project_root),
        "research_ready": coverage["research_ready"],
    }

    target = output_root / config.name
    if target.exists():
        raise FileExistsError(f"report output already exists: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{config.name}-", dir=output_root))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "report.md").write_text(markdown, encoding="utf-8")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_source_snapshot(project_root, temporary / "source.zip")
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "report.md"
