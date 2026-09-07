"""Tests for the strict Stage 13 evidence catalog."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from uav_swarm_control.configuration import (
    ConfigurationError,
    EvidenceStatus,
    cross_paper_config_from_mapping,
    load_cross_paper_config,
)

ROOT = Path(__file__).parents[2]


def test_stage13_catalog_requires_every_completion_gate_field() -> None:
    config = load_cross_paper_config(ROOT / "configs/comparison/stage13_cross_paper.yaml")

    assert len(config.native_systems) == 5
    assert config.native_systems[0].environment.status is EvidenceStatus.REPORTED
    assert config.native_systems[2].simulator_version.status is EvidenceStatus.MEASURED
    assert config.native_systems[-1].training_budget.status is EvidenceStatus.NOT_APPLICABLE
    assert config.expected_tasks == ("paper04-3uav", "paper04-4uav", "paper04-5uav")


def test_catalog_rejects_missing_uncertainty_field() -> None:
    value = yaml.safe_load(
        (ROOT / "configs/comparison/stage13_cross_paper.yaml").read_text(encoding="utf-8")
    )
    broken = deepcopy(value)
    del broken["native_systems"][0]["uncertainty"]

    with pytest.raises(ConfigurationError, match="missing required keys: uncertainty"):
        cross_paper_config_from_mapping(broken)


def test_catalog_rejects_unknown_evidence_source() -> None:
    value = yaml.safe_load(
        (ROOT / "configs/comparison/stage13_cross_paper.yaml").read_text(encoding="utf-8")
    )
    broken = deepcopy(value)
    broken["native_systems"][0]["environment"]["source"] = "missing: page 1"

    with pytest.raises(ConfigurationError, match="unknown source"):
        cross_paper_config_from_mapping(broken)
