"""Tests for deterministic simulator-independent obstacle mechanics."""

import numpy as np

from uav_swarm_control.configuration import load_obstacle_experiment_config
from uav_swarm_control.obstacles import (
    ObstacleField,
    ObstacleScenario,
    advance_obstacles,
    obstacle_safety,
    sample_obstacle_field,
)


def test_constant_velocity_motion_and_surface_clearance() -> None:
    field = ObstacleField(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[-0.2, 0.0, 0.0]]),
        np.array([0.15]),
    )
    advanced = advance_obstacles(field, 0.5)
    safety = obstacle_safety([[0.0, 0.0, 0.0]], advanced, vehicle_radius_m=0.05)

    np.testing.assert_allclose(advanced.positions_m, [[0.9, 0.0, 0.0]])
    np.testing.assert_allclose(safety.clearances_m, [[0.7]])
    assert safety.minimum_clearance_m == 0.7
    assert safety.collision_pairs == 0


def test_collision_uses_strict_sphere_overlap() -> None:
    field = ObstacleField(np.array([[0.1, 0.0, 0.0]]), np.zeros((1, 3)), np.array([0.06]))
    safety = obstacle_safety([[0.0, 0.0, 0.0]], field, vehicle_radius_m=0.05)

    assert safety.collision_pairs == 1
    assert safety.collided_agents.tolist() == [True]
    assert safety.minimum_clearance_m is not None
    assert safety.minimum_clearance_m < 0.0


def test_sampling_is_seeded_and_respects_scenario_counts() -> None:
    config = load_obstacle_experiment_config(
        "configs/experiment/stage10_dynamic_obstacles_4uav.yaml"
    )
    origins = np.array([[-1.4, -0.3, 1.6], [-1.4, 0.3, 1.6]])
    targets = np.array([[0.0, -0.3, 2.0], [0.0, 0.3, 2.0]])

    for scenario in ObstacleScenario:
        first = sample_obstacle_field(
            scenario, origins, targets, config.field, np.random.default_rng(7)
        )
        second = sample_obstacle_field(
            scenario, origins, targets, config.field, np.random.default_rng(7)
        )
        assert first.count == config.field.count_for(scenario)
        np.testing.assert_array_equal(first.positions_m, second.positions_m)
        np.testing.assert_array_equal(first.velocities_mps, second.velocities_mps)
        initial = obstacle_safety(origins, first, vehicle_radius_m=config.field.vehicle_radius_m)
        assert initial.collision_pairs == 0
        if scenario is ObstacleScenario.STATIC:
            assert first.dynamic_count == 0
        if scenario is ObstacleScenario.SLOW_DYNAMIC:
            assert first.dynamic_count == first.count
