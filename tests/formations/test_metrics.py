"""Behavioral tests for fixed-assignment formation metrics."""

import math

import numpy as np
import pytest

from uav_swarm_control.formations import (
    normalized_shape_mse,
    normalized_shape_rmse,
    position_mse,
    position_rmse,
    position_sse,
    rigid_alignment,
    rotation_matrix_from_euler,
    shape_mse,
    shape_rmse,
    shape_sse,
    square_pyramid,
    transform,
    triangle,
)


def test_position_metrics_have_explicit_aggregation() -> None:
    """SSE, MSE, and RMSE should not be confused."""
    current = np.zeros((2, 3))
    target = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    assert position_sse(current, target) == pytest.approx(25.0)
    assert position_mse(current, target) == pytest.approx(12.5)
    assert position_rmse(current, target) == pytest.approx(math.sqrt(12.5))


def test_position_mse_is_normalized_by_agent_count() -> None:
    """Repeating equivalent agents must not change mean error."""
    current = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = np.zeros((2, 3))

    assert position_mse(np.tile(current, (4, 1)), np.tile(target, (4, 1))) == pytest.approx(
        position_mse(current, target)
    )


def test_shape_error_ignores_translation_and_rotation() -> None:
    """Rigidly moving a correct formation must preserve its shape score."""
    target = square_pyramid(spacing=0.5)
    rotation = rotation_matrix_from_euler(roll=0.4, pitch=-0.2, yaw=1.1)
    current = transform(target, rotation=rotation, offset=(2.0, -3.0, 1.0))

    assert shape_sse(current, target) == pytest.approx(0.0, abs=1e-24)
    assert shape_mse(current, target) == pytest.approx(0.0, abs=1e-24)
    assert shape_rmse(current, target) == pytest.approx(0.0, abs=1e-12)
    assert position_rmse(current, target) > 0.0


def test_shape_error_preserves_scale_error() -> None:
    """A formation with the wrong physical spacing is not a correct shape."""
    target = triangle(spacing=0.5)
    current = target * 2.0

    assert shape_rmse(current, target) > 0.0


def test_normalized_shape_error_is_dimensionless() -> None:
    """Changing all physical units should not change normalized error."""
    target = square_pyramid(spacing=0.5)
    current = target.copy()
    current[0, 0] += 0.1

    original = normalized_shape_mse(current, target)
    rescaled = normalized_shape_mse(current * 100.0, target * 100.0)

    assert rescaled == pytest.approx(original)
    assert normalized_shape_rmse(current, target) == pytest.approx(math.sqrt(original))


def test_alignment_returns_proper_rotation() -> None:
    """Kabsch alignment must not return a reflection."""
    target = square_pyramid(spacing=0.5)
    current = transform(
        target,
        rotation=rotation_matrix_from_euler(roll=0.2, pitch=0.4, yaw=-0.8),
        offset=(1.0, 2.0, 3.0),
    )

    alignment = rigid_alignment(current, target)

    assert np.linalg.det(alignment.rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(alignment.aligned_source, target, atol=1e-12)


def test_fixed_correspondence_penalizes_asymmetric_permutation() -> None:
    """Rows represent agent assignments; arbitrary reassignment is not hidden."""
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    current = target[[1, 0, 2, 3]]

    assert shape_mse(current, target) > 1e-6


def test_single_agent_has_zero_shape_error() -> None:
    """One point has pose but no internal formation shape."""
    assert normalized_shape_mse([[10.0, -2.0, 4.0]], [[1.0, 3.0, 8.0]]) == 0.0


def test_mismatched_agent_counts_are_rejected() -> None:
    """Metrics require one target per agent."""
    with pytest.raises(ValueError, match="identical shapes"):
        position_mse(np.zeros((2, 3)), np.zeros((3, 3)))


def test_coincident_target_cannot_normalize_shape_error() -> None:
    """A zero-diameter multi-agent target has no normalization length."""
    with pytest.raises(ValueError, match="distinct points"):
        normalized_shape_mse(np.zeros((2, 3)), np.zeros((2, 3)))
