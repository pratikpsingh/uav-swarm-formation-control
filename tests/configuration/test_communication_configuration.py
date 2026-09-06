"""Validation tests for the Stage 11 topology protocol."""

from pathlib import Path
from typing import cast

import pytest
import yaml

from uav_swarm_control.communication import CommunicationRegimenKind
from uav_swarm_control.configuration import (
    ConfigurationError,
    load_communication_experiment_config,
)
from uav_swarm_control.evaluation.communication import student_t_critical_95

ROOT = Path(__file__).parents[2]
CONFIGS = (
    ROOT / "configs/experiment/stage11_plane_4uav.yaml",
    ROOT / "configs/experiment/stage11_pyramid_5uav.yaml",
)


def test_protocol_varies_formation_size_and_topology_factors() -> None:
    configs = [load_communication_experiment_config(path) for path in CONFIGS]

    assert [
        (item.mappo.experiment.formation.kind.value, item.mappo.experiment.formation.num_agents)
        for item in configs
    ] == [
        ("plane", 4),
        ("pyramid", 5),
    ]
    for config in configs:
        assert len(config.training_seeds) == 5
        assert len(config.evaluation_conditions) == 16
        assert (
            sum(
                regimen.kind is CommunicationRegimenKind.FIXED
                for regimen in config.training_regimens
            )
            == 2
        )
        variable = config.training_regimens[-1]
        assert variable.kind is CommunicationRegimenKind.VARIABLE
        assert set(variable.conditions) == set(config.evaluation_conditions)
        assert config.encoder.payload_bytes_per_neighbor == 24


def test_five_seed_interval_uses_the_student_t_critical_value() -> None:
    assert student_t_critical_95(5) == pytest.approx(2.7764451051977987)


def test_missing_factor_level_is_rejected(tmp_path: Path) -> None:
    raw = cast(
        dict[str, object],
        yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8")),
    )
    study = cast(dict[str, object], raw["communication_study"])
    conditions = cast(list[dict[str, object]], study["evaluation_conditions"])
    conditions[:] = [condition for condition in conditions if condition["sensing_radius_m"] is None]
    regimens = cast(list[dict[str, object]], study["training_regimens"])
    regimens[-1]["conditions"] = [condition["name"] for condition in conditions]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must vary neighbor count"):
        load_communication_experiment_config(path)
