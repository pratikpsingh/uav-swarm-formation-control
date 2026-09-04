"""Tests for reward-independent kinematic measurements."""

import numpy as np
import pytest

from uav_swarm_control.evaluation import collision_statistics, evaluate_kinematic_state


def test_collision_statistics_count_unique_pairs_and_agents() -> None:
    statistics = collision_statistics(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        collision_distance_m=0.2,
    )

    assert statistics.pair_count == 1
    assert statistics.minimum_separation_m == pytest.approx(0.1)
    np.testing.assert_array_equal(statistics.collided_agents, [True, True, False])


def test_state_metrics_have_hand_calculable_position_and_control_errors() -> None:
    targets = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    actions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    previous_actions = np.zeros((2, 3))

    metrics = evaluate_kinematic_state(
        positions,
        targets,
        actions,
        previous_actions,
        collision_distance_m=0.2,
        success_tolerance_m=0.1,
    )

    assert metrics.position_rmse_m == pytest.approx(np.sqrt(0.5))
    assert metrics.shape_rmse_m == pytest.approx(0.5)
    assert metrics.normalized_shape_rmse == pytest.approx(0.25)
    assert metrics.centroid_error_m == pytest.approx(0.5)
    assert metrics.minimum_separation_m == pytest.approx(1.0)
    assert metrics.control_delta_rms == pytest.approx(np.sqrt(1.0 / 6.0))
    assert not metrics.within_tolerance


def test_success_requires_tolerance_and_no_collision() -> None:
    targets = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    zeros = np.zeros((2, 3))

    metrics = evaluate_kinematic_state(
        targets,
        targets,
        zeros,
        zeros,
        collision_distance_m=0.2,
        success_tolerance_m=0.01,
    )

    assert metrics.position_rmse_m == 0.0
    assert not metrics.within_tolerance
