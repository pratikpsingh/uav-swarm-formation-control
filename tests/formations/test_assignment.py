"""Tests for deterministic agent-to-target assignment."""

import itertools

import numpy as np
import pytest

from uav_swarm_control.formations import fixed_assignment, minimum_distance_assignment


def test_fixed_assignment_preserves_rows_and_reports_distance() -> None:
    origins = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    targets = np.array([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assignment = fixed_assignment(origins, targets)

    assert assignment.target_indices == (0, 1)
    np.testing.assert_array_equal(assignment.assigned_targets, targets)
    assert assignment.total_distance_m == pytest.approx(4.0)


def test_minimum_distance_assignment_recovers_a_permutation() -> None:
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    targets = origins[[2, 0, 1]]

    assignment = minimum_distance_assignment(origins, targets)

    assert assignment.target_indices == (1, 2, 0)
    np.testing.assert_array_equal(assignment.assigned_targets, origins)
    assert assignment.total_distance_m == 0.0


def test_hungarian_result_matches_exhaustive_optimum() -> None:
    rng = np.random.default_rng(20260909)
    origins = rng.normal(size=(5, 3))
    targets = rng.normal(size=(5, 3))
    costs = np.linalg.norm(origins[:, None, :] - targets[None, :, :], axis=2)

    assignment = minimum_distance_assignment(origins, targets)
    exhaustive = min(
        sum(costs[row, column] for row, column in enumerate(permutation))
        for permutation in itertools.permutations(range(5))
    )

    assert assignment.total_distance_m == pytest.approx(exhaustive)
    assert sorted(assignment.target_indices) == list(range(5))


def test_assignment_rejects_mismatched_point_sets() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        minimum_distance_assignment(np.zeros((2, 3)), np.zeros((3, 3)))
