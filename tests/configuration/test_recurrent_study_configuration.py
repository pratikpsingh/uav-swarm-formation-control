"""Configuration tests for the integrated recurrent research protocol."""

from dataclasses import replace
from pathlib import Path

import pytest

from uav_swarm_control.configuration import (
    ConfigurationError,
    MissionStudyConfig,
    load_recurrent_study_config,
)
from uav_swarm_control.evaluation.recurrent_study import recurrent_study_smoke_config
from uav_swarm_control.formations import FormationKind

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiment/recurrent-study/sphere-8-uav.yaml"


def test_recurrent_study_configuration_covers_requested_capabilities() -> None:
    config = load_recurrent_study_config(CONFIG)

    assert config.communication.mappo.experiment.formation.num_agents == 8
    assert config.formation_kinds == (
        FormationKind.PLANE,
        FormationKind.PYRAMID,
        FormationKind.CUBE,
        FormationKind.SPHERE,
    )
    scenarios = {
        condition.obstacle_scenario for condition in config.communication.evaluation_conditions
    }
    counts = {
        condition.requested_neighbors for condition in config.communication.evaluation_conditions
    }
    assert {0, 1, 2, 3, 4, 6, 7} <= counts
    assert len(scenarios) == 4
    assert config.mission.initial_ground_altitude_m == 0.08

    smoke = recurrent_study_smoke_config(config)
    assert smoke.communication.profile == "smoke"
    assert smoke.recurrent.mappo.ppo.total_steps == 128
    assert smoke.recurrent.sequence_length == 8
    assert smoke.recovery.disturbance_step == 16
    assert (
        smoke.recovery.disturbance_step
        < smoke.communication.mappo.experiment.environment.max_episode_steps
    )


def test_construction_altitude_must_be_above_ground() -> None:
    with pytest.raises(ConfigurationError, match="mission altitudes"):
        MissionStudyConfig(
            initial_ground_altitude_m=1.0,
            construction_altitude_m=0.5,
            construction_hold_steps=2,
            waypoint_spacing_m=1.0,
            waypoint_hold_steps=2,
        )

    config = load_recurrent_study_config(CONFIG)
    with pytest.raises(ConfigurationError, match="mission altitudes"):
        replace(
            config.mission,
            initial_ground_altitude_m=config.mission.construction_altitude_m,
        )
