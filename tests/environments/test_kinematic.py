"""Tests for the deterministic 3D point-mass environment."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_swarm_control.agents import AgentId
from uav_swarm_control.configuration import ExperimentConfig, load_experiment_config
from uav_swarm_control.controllers import ProportionalPositionController
from uav_swarm_control.environments import (
    MultiAgentEnvironment,
    NormalizedVelocityActions,
)
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment
from uav_swarm_control.evaluation import run_episode


def _config() -> ExperimentConfig:
    path = Path(__file__).parents[2] / "configs" / "experiment" / "triangle_kinematic.yaml"
    return load_experiment_config(path)


def _controller(config: ExperimentConfig) -> ProportionalPositionController:
    return ProportionalPositionController(
        gain_per_second=config.controller.gain_per_second,
        max_velocity_component_mps=config.environment.max_velocity_component_mps,
    )


def test_reset_is_seeded_and_satisfies_contract_shapes() -> None:
    config = _config()
    environment = KinematicSwarmEnvironment(config)

    first = environment.reset(seed=123)
    first_positions = environment.positions
    second = environment.reset(seed=123)

    np.testing.assert_array_equal(first_positions, environment.positions)
    assert isinstance(environment, MultiAgentEnvironment)
    assert first.observations.ego.shape == (3, 6)
    assert first.observations.neighbors.shape == (3, 2, 6)
    assert first.observations.neighbor_mask.all()
    assert first.centralized_state.values.shape == (27,)
    np.testing.assert_allclose(
        second.observations.ego[:, :3],
        environment.target_positions - environment.positions,
        atol=1e-6,
    )


def test_different_reset_seeds_change_noisy_initial_positions() -> None:
    environment = KinematicSwarmEnvironment(_config())
    environment.reset(seed=123)
    first = environment.positions
    environment.reset(seed=456)

    assert not np.array_equal(first, environment.positions)


def test_neighbors_are_selected_by_distance_then_agent_id() -> None:
    config = _config()
    config = replace(
        config,
        observation=replace(config.observation, max_neighbors=1),
        task=replace(config.task, initial_position_noise_m=0.0),
    )
    environment = KinematicSwarmEnvironment(config)

    observations = environment.reset(seed=1).observations
    positions = environment.positions

    expected_neighbor_indices = (1, 0, 0)
    for agent_index, neighbor_index in enumerate(expected_neighbor_indices):
        np.testing.assert_allclose(
            observations.neighbors[agent_index, 0, :3],
            positions[neighbor_index] - positions[agent_index],
            atol=1e-6,
        )


def test_neighbor_slots_are_masked_outside_sensing_radius() -> None:
    config = _config()
    config = replace(
        config,
        observation=replace(config.observation, neighbor_radius_m=0.5),
        task=replace(config.task, initial_position_noise_m=0.0),
    )
    environment = KinematicSwarmEnvironment(config)

    observations = environment.reset(seed=1).observations

    assert not observations.neighbor_mask.any()
    np.testing.assert_array_equal(observations.neighbors, 0.0)


def test_step_applies_first_order_velocity_dynamics() -> None:
    config = _config()
    environment = KinematicSwarmEnvironment(config)
    environment.reset(seed=123)
    before = environment.positions
    action_values = np.full((3, 3), 0.5, dtype=np.float32)

    result = environment.step(NormalizedVelocityActions(environment.agent_ids, action_values))

    expected_delta = (
        0.5 * config.environment.max_velocity_component_mps * config.environment.time_step_seconds
    )
    np.testing.assert_allclose(environment.positions - before, expected_delta)
    assert result.metrics["step_count"] == 1.0


def test_time_limit_is_reported_as_truncation() -> None:
    config = _config()
    config = replace(
        config,
        environment=replace(config.environment, max_episode_steps=1),
        task=replace(config.task, initial_position_noise_m=0.0),
    )
    environment = KinematicSwarmEnvironment(config)
    environment.reset(seed=config.seed)
    actions = NormalizedVelocityActions(
        environment.agent_ids,
        np.zeros((3, 3), dtype=np.float32),
    )

    result = environment.step(actions)

    assert result.truncated
    assert not result.terminated
    assert not result.metrics["success"]


def test_actions_must_preserve_environment_agent_order() -> None:
    environment = KinematicSwarmEnvironment(_config())
    environment.reset(seed=1)
    wrong_ids = (AgentId(1), AgentId(0), AgentId(2))

    with pytest.raises(ValueError, match="row order"):
        environment.step(
            NormalizedVelocityActions(
                wrong_ids,
                np.zeros((3, 3), dtype=np.float32),
            )
        )


def test_scripted_controller_solves_seeded_triangle_task_deterministically() -> None:
    config = _config()

    first = run_episode(
        KinematicSwarmEnvironment(config),
        _controller(config),
        seed=config.seed,
        safety_step_limit=config.environment.max_episode_steps,
    )
    second = run_episode(
        KinematicSwarmEnvironment(config),
        _controller(config),
        seed=config.seed,
        safety_step_limit=config.environment.max_episode_steps,
    )

    assert first.success
    assert first.terminated and not first.truncated
    assert first.steps < config.environment.max_episode_steps
    assert first.final_metrics["position_rmse_m"] <= config.task.success_tolerance_m
    assert first.final_metrics["collision_pairs"] == 0.0
    assert first.final_metrics["success_streak"] == float(config.task.success_hold_steps)
    assert second.steps == first.steps
    np.testing.assert_array_equal(second.returns, first.returns)


def test_controller_solves_tilted_3d_triangle_task() -> None:
    config = _config()
    config = replace(
        config,
        formation=replace(
            config.formation,
            center_m=(0.5, -0.5, 2.0),
            euler_radians=(0.3, -0.2, 0.4),
        ),
        task=replace(config.task, initial_center_m=(-1.5, 0.5, 0.5)),
    )

    result = run_episode(
        KinematicSwarmEnvironment(config),
        _controller(config),
        seed=config.seed,
        safety_step_limit=config.environment.max_episode_steps,
    )

    assert result.success
    assert result.final_metrics["position_rmse_m"] <= config.task.success_tolerance_m
