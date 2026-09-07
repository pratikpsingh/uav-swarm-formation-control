"""Tests for guarded common-protocol comparisons."""

import json
from pathlib import Path

import pytest

from uav_swarm_control.configuration.baseline import load_baseline_config
from uav_swarm_control.evaluation.baseline import smoke_config
from uav_swarm_control.evaluation.benchmark import summarize_records
from uav_swarm_control.evaluation.comparison import (
    compare_controller_results,
    comparison_protocol,
)
from uav_swarm_control.evaluation.dmpc import dmpc_smoke_config

ROOT = Path(__file__).parents[2]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_episode_summary_labels_its_sampling_unit() -> None:
    summary = summarize_records([{"success": 0.0}, {"success": 1.0}], count_key="episode_count")
    assert summary["episode_count"] == 2
    assert "seed_count" not in summary


def test_comparison_retains_different_uncertainty_units(tmp_path: Path) -> None:
    mappo = tmp_path / "mappo.json"
    dmpc = tmp_path / "dmpc.json"
    output = tmp_path / "comparison.json"
    _write(
        mappo,
        {
            "comparison_fingerprint": "same",
            "metrics": {"success": {"mean": 0.8, "sample_std": 0.1}},
        },
    )
    _write(
        dmpc,
        {
            "comparison_fingerprint": "same",
            "common_summary": {"metrics": {"success": {"mean": 1.0, "sample_std": 0.0}}},
        },
    )

    compare_controller_results(mappo, dmpc, output)

    result = json.loads(output.read_text())
    assert result["metrics"]["success"] == {
        "mappo_mean": 0.8,
        "mappo_sample_std_across_training_seeds": 0.1,
        "dmpc_mean": 1.0,
        "dmpc_sample_std_across_evaluation_episodes": 0.0,
    }
    assert "not equivalent" in result["warning"]


def test_comparison_rejects_different_environment_protocols(tmp_path: Path) -> None:
    mappo = tmp_path / "mappo.json"
    dmpc = tmp_path / "dmpc.json"
    _write(mappo, {"comparison_fingerprint": "one", "metrics": {}})
    _write(dmpc, {"comparison_fingerprint": "two", "common_summary": {"metrics": {}}})
    with pytest.raises(ValueError, match="different comparison protocols"):
        compare_controller_results(mappo, dmpc, tmp_path / "comparison.json")


def test_smoke_profiles_have_identical_comparison_protocols() -> None:
    config = load_baseline_config(ROOT / "configs/experiment/baseline/triangle-3-uav.yaml")
    assert comparison_protocol(smoke_config(config)) == comparison_protocol(
        dmpc_smoke_config(config)
    )
