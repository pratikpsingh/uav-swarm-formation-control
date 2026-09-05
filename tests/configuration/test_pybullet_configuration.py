"""Tests for simulator-specific configuration and timing invariants."""

from dataclasses import replace
from pathlib import Path

import pytest

from uav_swarm_control.configuration import (
    ConfigurationError,
    PyBulletExperimentConfig,
    PyBulletSimulatorConfig,
    load_pybullet_experiment_config,
    pybullet_experiment_config_to_dict,
)

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "experiment"


def test_loads_resolved_pybullet_configuration() -> None:
    config = load_pybullet_experiment_config(CONFIG_DIRECTORY / "pybullet_hover.yaml")

    assert config.simulator.physics_steps_per_control == 5
    assert config.experiment.environment.time_step_seconds == pytest.approx(1.0 / 48.0)
    assert pybullet_experiment_config_to_dict(config)["simulator"] == {
        "drone_model": "cf2x",
        "physics": "pyb",
        "physics_frequency_hz": 240,
        "control_frequency_hz": 48,
        "physics_steps_per_control": 5,
        "gui": False,
        "record_video": False,
    }


def test_rejects_inconsistent_control_timestep() -> None:
    config = load_pybullet_experiment_config(CONFIG_DIRECTORY / "pybullet_hover.yaml")
    inconsistent_experiment = replace(
        config.experiment,
        environment=replace(config.experiment.environment, time_step_seconds=0.1),
    )

    with pytest.raises(ConfigurationError, match="must equal 1 / simulator"):
        PyBulletExperimentConfig(inconsistent_experiment, config.simulator)


def test_rejects_non_integral_physics_ratio() -> None:
    config = load_pybullet_experiment_config(CONFIG_DIRECTORY / "pybullet_hover.yaml")

    with pytest.raises(ConfigurationError, match="must be divisible"):
        PyBulletSimulatorConfig(
            drone_model=config.simulator.drone_model,
            physics=config.simulator.physics,
            physics_frequency_hz=240,
            control_frequency_hz=50,
            gui=False,
            record_video=False,
        )
