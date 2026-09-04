"""Behavioral tests for 3D formation transforms."""

import math

import numpy as np
import pytest

from uav_swarm_control.formations import (
    center,
    pairwise_distances,
    rotate,
    rotation_matrix_from_euler,
    scale,
    transform,
    translate,
    triangle,
    validate_rotation_matrix,
)


def test_center_moves_centroid_without_mutating_input() -> None:
    """Centering returns a new array and leaves measurements untouched."""
    points = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    original = points.copy()

    centered = center(points)

    np.testing.assert_allclose(centered.mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_array_equal(points, original)


def test_translation_moves_centroid_without_changing_shape() -> None:
    """Translation changes pose, not internal geometry."""
    points = triangle(0.5)
    offset = np.array([2.0, -1.0, 3.0])

    moved = translate(points, offset)

    np.testing.assert_allclose(moved.mean(axis=0), offset, atol=1e-12)
    np.testing.assert_allclose(pairwise_distances(moved), pairwise_distances(points))


def test_scaling_multiplies_pairwise_distances() -> None:
    """Scale has a predictable physical interpretation."""
    points = triangle(0.5)

    enlarged = scale(points, 3.0)

    np.testing.assert_allclose(
        pairwise_distances(enlarged),
        3.0 * pairwise_distances(points),
    )


def test_roll_rotates_y_axis_to_z_axis() -> None:
    """Euler angles use the documented active right-handed convention."""
    rotation = rotation_matrix_from_euler(roll=math.pi / 2.0)

    rotated = rotate(np.array([[0.0, 1.0, 0.0]]), rotation)

    np.testing.assert_allclose(rotated, np.array([[0.0, 0.0, 1.0]]), atol=1e-12)


def test_arbitrary_3d_rotation_preserves_pairwise_distances() -> None:
    """Roll, pitch, and yaw must not deform a formation."""
    points = triangle(0.5)
    rotation = rotation_matrix_from_euler(roll=0.3, pitch=-0.7, yaw=1.2)

    rotated = rotate(points, rotation)

    np.testing.assert_allclose(pairwise_distances(rotated), pairwise_distances(points))


def test_transform_applies_scale_rotation_then_translation() -> None:
    """The transformation order is explicit and stable."""
    points = np.array([[1.0, 0.0, 0.0]])
    rotation = rotation_matrix_from_euler(yaw=math.pi / 2.0)

    transformed = transform(
        points,
        scale_factor=2.0,
        rotation=rotation,
        offset=(1.0, 2.0, 3.0),
    )

    np.testing.assert_allclose(transformed, np.array([[1.0, 4.0, 3.0]]), atol=1e-12)


def test_reflection_is_not_accepted_as_rotation() -> None:
    """A reflection changes handedness and is not an SO(3) rotation."""
    reflection = np.diag([-1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="determinant"):
        validate_rotation_matrix(reflection)


def test_non_orthogonal_matrix_is_rejected() -> None:
    """Invalid matrices should fail before corrupting geometry."""
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rotation_matrix(np.diag([1.0, 1.0, 2.0]))


def test_non_positive_scale_is_rejected() -> None:
    """A positive scale avoids collapse and implicit reflection."""
    with pytest.raises(ValueError, match="factor"):
        scale(triangle(), 0.0)
