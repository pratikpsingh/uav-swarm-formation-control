"""Tests for two-track Stage 13 report normalization."""

import hashlib
import json
from pathlib import Path

import pytest

from uav_swarm_control.configuration import (
    ControlledMethod,
    CrossPaperComparisonConfig,
    EvidenceSource,
    EvidenceStatus,
    EvidenceValue,
    NativeSystemEvidence,
)
from uav_swarm_control.evaluation.cross_paper import (
    normalize_controlled_comparisons,
    render_markdown_report,
    verify_evidence_sources,
)


def _config(source: EvidenceSource) -> CrossPaperComparisonConfig:
    evidence = EvidenceValue(EvidenceStatus.REPORTED, "value", "paper: test")
    native = NativeSystemEvidence(
        "native",
        "Native",
        "test method",
        evidence,
        evidence,
        evidence,
        evidence,
        evidence,
        evidence,
        evidence,
        evidence,
        evidence,
        evidence,
        "not comparable",
    )
    methods = (
        ControlledMethod("mappo", "MAPPO", "velocity", "local", "decentralized"),
        ControlledMethod("dmpc", "DMPC", "trajectory", "shared state", "per-UAV optimizer"),
    )
    return CrossPaperComparisonConfig(
        "test-report",
        (source,),
        (native,),
        methods,
        ("task-a",),
        ("success",),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _controlled_fixture(
    tmp_path: Path, *, fingerprint: str = "same"
) -> tuple[Path, EvidenceSource]:
    evidence = tmp_path / "paper.pdf"
    evidence.write_bytes(b"paper")
    source = EvidenceSource(
        "paper",
        "Paper",
        Path("paper.pdf"),
        hashlib.sha256(b"paper").hexdigest(),
    )
    task = tmp_path / "task"
    summary = task / "summary.json"
    result = task / "dmpc" / "result.json"
    comparison = task / "comparison.json"
    protocol = {
        "fingerprint": fingerprint,
        "physics": {
            "experiment": {
                "name": "task-a",
                "formation": {"kind": "plane", "num_agents": 4},
                "environment": {"max_episode_steps": 48},
            },
            "simulator": {
                "physics": "pyb",
                "control_frequency_hz": 30,
            },
        },
    }
    _write_json(
        task / "manifest.json",
        {
            "method": "mappo",
            "comparison_protocol": protocol,
            "configuration": {"mappo": {"experiment": {"name": "task-a"}}},
        },
    )
    _write_json(
        summary,
        {
            "comparison_fingerprint": fingerprint,
            "profile": "smoke",
            "seed_count": 2,
            "evaluation_episodes_per_seed": 3,
            "evaluation_root_seed": 9,
            "metrics": {"training_environment_steps": {"mean": 256.0}},
        },
    )
    _write_json(
        result,
        {
            "comparison_fingerprint": fingerprint,
            "profile": "smoke",
            "method": "dmpc",
            "common_summary": {"episode_count": 3},
            "simulator": {
                "simulator_package": "sim",
                "simulator_version": "1.0",
                "simulator_revision": "abc",
                "pybullet_api_version": 1,
            },
        },
    )
    _write_json(
        comparison,
        {
            "artifact_schema_version": 1,
            "comparison_fingerprint": fingerprint,
            "mappo_result": str(summary),
            "dmpc_result": str(result),
            "metrics": {
                "success": {
                    "mappo_mean": 0.5,
                    "mappo_sample_std_across_training_seeds": 0.1,
                    "dmpc_mean": 0.75,
                    "dmpc_sample_std_across_evaluation_episodes": 0.2,
                }
            },
        },
    )
    return comparison, source


def test_source_verification_rejects_changed_evidence(tmp_path: Path) -> None:
    comparison, source = _controlled_fixture(tmp_path)
    del comparison
    config = _config(source)
    (tmp_path / "paper.pdf").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum changed"):
        verify_evidence_sources(config, tmp_path)


def test_controlled_report_retains_context_and_uncertainty_units(tmp_path: Path) -> None:
    comparison, source = _controlled_fixture(tmp_path)
    config = _config(source)
    rows, coverage = normalize_controlled_comparisons(config, [comparison], project_root=tmp_path)

    assert coverage["research_ready"] is False
    assert len(rows) == 2
    assert rows[0]["training_budget"] == "256 environment transitions per seed"
    assert "trained seed means" in str(rows[0]["uncertainty"])
    assert "evaluation episodes" in str(rows[1]["uncertainty"])


def test_controlled_report_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    comparison, source = _controlled_fixture(tmp_path)
    value = json.loads(comparison.read_text(encoding="utf-8"))
    value["comparison_fingerprint"] = "different"
    comparison.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        normalize_controlled_comparisons(_config(source), [comparison], project_root=tmp_path)


def test_markdown_controlled_table_carries_completion_gate_columns(tmp_path: Path) -> None:
    comparison, source = _controlled_fixture(tmp_path)
    config = _config(source)
    rows, coverage = normalize_controlled_comparisons(config, [comparison], project_root=tmp_path)
    markdown = render_markdown_report(config, [], rows, coverage, [])

    header = next(line for line in markdown.splitlines() if line.startswith("| Track"))
    for heading in (
        "Environment",
        "Simulator version",
        "Action semantics",
        "Observation access",
        "Training budget",
        "Seeds",
        "Uncertainty",
        "Execution",
    ):
        assert heading in header
