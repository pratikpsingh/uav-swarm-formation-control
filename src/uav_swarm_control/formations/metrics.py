"""Fixed-assignment position and formation-shape metrics."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations._validation import matching_point_arrays, points_array


@dataclass(frozen=True, slots=True)
class RigidAlignment:
    """Least-squares rigid alignment of source points to target points."""

    rotation: FloatArray
    translation: FloatArray
    aligned_source: FloatArray


def pairwise_distances(points: ArrayLike) -> FloatArray:
    """Return the Euclidean distance matrix for a point set."""
    point_array = points_array(points, name="points")
    differences = point_array[:, None, :] - point_array[None, :, :]
    return np.linalg.norm(differences, axis=2)


def formation_diameter(points: ArrayLike) -> float:
    """Return the largest pairwise distance, or zero for one point."""
    return float(pairwise_distances(points).max())


def rigid_alignment(source: ArrayLike, target: ArrayLike) -> RigidAlignment:
    """Align source to target using translation and a proper Kabsch rotation.

    Row correspondence is fixed: source row i aligns with target row i.
    Scale and reflection are not fitted.
    """
    source_array, target_array = matching_point_arrays(source, target)
    source_centroid = source_array.mean(axis=0)
    target_centroid = target_array.mean(axis=0)
    source_centered = source_array - source_centroid
    target_centered = target_array - target_centroid

    covariance = source_centered.T @ target_centered
    left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    candidate = right_vectors_transposed.T @ left_vectors.T
    if np.linalg.det(candidate) < 0.0:
        correction[-1, -1] = -1.0
    rotation = right_vectors_transposed.T @ correction @ left_vectors.T
    translation = target_centroid - source_centroid @ rotation.T
    aligned_source = source_array @ rotation.T + translation
    return RigidAlignment(
        rotation=rotation,
        translation=translation,
        aligned_source=aligned_source,
    )


def squared_position_errors(current: ArrayLike, target: ArrayLike) -> FloatArray:
    """Return one squared Euclidean target error per assigned agent."""
    current_array, target_array = matching_point_arrays(current, target)
    return np.sum((current_array - target_array) ** 2, axis=1)


def position_sse(current: ArrayLike, target: ArrayLike) -> float:
    """Return the sum of squared assigned-position errors."""
    return float(squared_position_errors(current, target).sum())


def position_mse(current: ArrayLike, target: ArrayLike) -> float:
    """Return the mean squared assigned-position error per agent."""
    return float(squared_position_errors(current, target).mean())


def position_rmse(current: ArrayLike, target: ArrayLike) -> float:
    """Return the root-mean-square assigned-position error in meters."""
    return math.sqrt(position_mse(current, target))


def squared_shape_errors(current: ArrayLike, target: ArrayLike) -> FloatArray:
    """Return per-agent residuals after optimal rigid alignment."""
    current_array, target_array = matching_point_arrays(current, target)
    alignment = rigid_alignment(current_array, target_array)
    return np.sum((alignment.aligned_source - target_array) ** 2, axis=1)


def shape_sse(current: ArrayLike, target: ArrayLike) -> float:
    """Return rigid-aligned shape sum of squared errors."""
    return float(squared_shape_errors(current, target).sum())


def shape_mse(current: ArrayLike, target: ArrayLike) -> float:
    """Return rigid-aligned mean squared shape error per agent."""
    return float(squared_shape_errors(current, target).mean())


def shape_rmse(current: ArrayLike, target: ArrayLike) -> float:
    """Return rigid-aligned root-mean-square shape error in meters."""
    return math.sqrt(shape_mse(current, target))


def normalized_shape_mse(current: ArrayLike, target: ArrayLike) -> float:
    """Return dimensionless shape MSE divided by squared target diameter."""
    current_array, target_array = matching_point_arrays(current, target)
    if len(target_array) == 1:
        return 0.0
    diameter = formation_diameter(target_array)
    if diameter <= np.finfo(np.float64).eps:
        raise ValueError("target must contain distinct points to normalize shape error.")
    return shape_mse(current_array, target_array) / diameter**2


def normalized_shape_rmse(current: ArrayLike, target: ArrayLike) -> float:
    """Return dimensionless shape RMSE divided by target diameter."""
    return math.sqrt(normalized_shape_mse(current, target))


__all__ = [
    "RigidAlignment",
    "formation_diameter",
    "normalized_shape_mse",
    "normalized_shape_rmse",
    "pairwise_distances",
    "position_mse",
    "position_rmse",
    "position_sse",
    "rigid_alignment",
    "shape_mse",
    "shape_rmse",
    "shape_sse",
    "squared_position_errors",
    "squared_shape_errors",
]
