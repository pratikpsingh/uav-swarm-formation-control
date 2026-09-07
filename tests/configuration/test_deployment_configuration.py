"""Policy-compression configuration invariants."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from uav_swarm_control.configuration import ConfigurationError
from uav_swarm_control.configuration.deployment import (
    deployment_smoke_config,
    deployment_study_config_from_mapping,
    load_deployment_study_config,
)
from uav_swarm_control.deployment.contracts import DeploymentArchitecture

CONFIG = Path("configs/deployment/policy-compression.yaml")


def test_research_matrix_covers_every_planned_comparison() -> None:
    config = load_deployment_study_config(CONFIG)
    assert {candidate.architecture for candidate in config.candidates} == set(
        DeploymentArchitecture
    )
    assert (
        len(
            {
                candidate.width
                for candidate in config.candidates
                if candidate.architecture is DeploymentArchitecture.FEED_FORWARD
                and candidate.quantization == "none"
                and candidate.structured_pruning_fraction == 0.0
            }
        )
        >= 3
    )
    assert any(candidate.structured_pruning_fraction > 0.0 for candidate in config.candidates)
    assert any(candidate.quantization == "int8-dynamic" for candidate in config.candidates)
    assert len(config.distillation_seeds) == 5


def test_smoke_profile_bounds_work_without_removing_factors_or_seeds() -> None:
    research = load_deployment_study_config(CONFIG)
    smoke = deployment_smoke_config(research)
    assert smoke.profile == "smoke"
    assert smoke.candidates == research.candidates
    assert smoke.distillation_seeds == research.distillation_seeds
    assert smoke.distillation.epochs < research.distillation.epochs


def test_research_rejects_too_few_independent_distillation_seeds() -> None:
    config = load_deployment_study_config(CONFIG)
    with pytest.raises(ConfigurationError, match="five distinct"):
        replace(config, distillation_seeds=(1, 2))


def test_closed_schema_rejects_unknown_keys() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["typo"] = True
    with pytest.raises(ConfigurationError, match="unknown keys"):
        deployment_study_config_from_mapping(cast(object, raw))
