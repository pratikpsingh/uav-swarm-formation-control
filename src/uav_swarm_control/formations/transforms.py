"""Pure transformations for formation point sets."""

import math

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations._validation import (
    finite_scalar,
    points_array,
    positive_scalar,
    vector3,
)

_ROTATION_TOLERANCE = 1e-9


def center(points: ArrayLike) -> FloatArray:
    """Return a copy whose centroid is the origin."""
    point_array = points_array(points, name="points")
    return point_array - point_array.mean(axis=0, keepdims=True)


def scale(points: ArrayLike, factor: float) -> FloatArray:
    """Scale points about the origin without modifying the input."""
    point_array = points_array(points, name="points")
    scale_factor = positive_scalar(factor, name="factor")
    return point_array * scale_factor


def translate(points: ArrayLike, offset: ArrayLike) -> FloatArray:
    """Translate points without modifying the input."""
    point_array = points_array(points, name="points")
    translation = vector3(offset, name="offset")
    return point_array + translation


def validate_rotation_matrix(rotation: ArrayLike) -> FloatArray:
    """Return a proper 3D rotation matrix or raise a descriptive error."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3); received {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("rotation must contain only finite values.")
    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=_ROTATION_TOLERANCE,
    ):
        raise ValueError("rotation must be orthonormal.")
    if not math.isclose(
        float(np.linalg.det(matrix)),
        1.0,
        rel_tol=0.0,
        abs_tol=_ROTATION_TOLERANCE,
    ):
        raise ValueError("rotation must be proper with determinant +1.")
    return matrix


def rotation_matrix_from_euler(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> FloatArray:
    """Create an active right-handed rotation from roll, pitch, and yaw.

    Rotations act on column vectors. Roll is applied first, then pitch,
    then yaw, giving R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll_angle = finite_scalar(roll, name="roll")
    pitch_angle = finite_scalar(pitch, name="pitch")
    yaw_angle = finite_scalar(yaw, name="yaw")

    cos_roll, sin_roll = math.cos(roll_angle), math.sin(roll_angle)
    cos_pitch, sin_pitch = math.cos(pitch_angle), math.sin(pitch_angle)
    cos_yaw, sin_yaw = math.cos(yaw_angle), math.sin(yaw_angle)

    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def rotate(points: ArrayLike, rotation: ArrayLike) -> FloatArray:
    """Apply an active rotation to row-wise points."""
    point_array = points_array(points, name="points")
    matrix = validate_rotation_matrix(rotation)
    return point_array @ matrix.T


def transform(
    points: ArrayLike,
    *,
    offset: ArrayLike = (0.0, 0.0, 0.0),
    rotation: ArrayLike | None = None,
    scale_factor: float = 1.0,
) -> FloatArray:
    """Scale, rotate, then translate a point set."""
    transformed = scale(points, scale_factor)
    if rotation is not None:
        transformed = rotate(transformed, rotation)
    return translate(transformed, offset)


__all__ = [
    "center",
    "rotate",
    "rotation_matrix_from_euler",
    "scale",
    "transform",
    "translate",
    "validate_rotation_matrix",
]
