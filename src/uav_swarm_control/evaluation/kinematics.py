"""Reward-independent measurements for kinematic swarm states."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_swarm_control.formations import (
    normalized_shape_rmse,
    pairwise_distances,
    position_rmse,
    shape_rmse,
)
from uav_swarm_control.formations._validation import matching_point_arrays, positive_scalar

type BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CollisionStatistics:
    """Pairwise collision information for one swarm state."""

    pair_count: int
    collided_agents: BoolArray
    minimum_separation_m: float


@dataclass(frozen=True, slots=True)
class KinematicStateMetrics:
    """Scientifically meaningful state and control measurements."""

    position_rmse_m: float
    shape_rmse_m: float
    normalized_shape_rmse: float
    centroid_error_m: float
    minimum_separation_m: float
    collision_pairs: int
    control_delta_rms: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, float]:
        """Return finite scalar metrics suitable for StepResult."""
        return {
            "position_rmse_m": self.position_rmse_m,
            "shape_rmse_m": self.shape_rmse_m,
            "normalized_shape_rmse": self.normalized_shape_rmse,
            "centroid_error_m": self.centroid_error_m,
            "minimum_separation_m": self.minimum_separation_m,
            "collision_pairs": float(self.collision_pairs),
            "control_delta_rms": self.control_delta_rms,
            "within_tolerance": float(self.within_tolerance),
        }


def collision_statistics(
    positions: ArrayLike,
    *,
    collision_distance_m: float,
) -> CollisionStatistics:
    """Count unique collision pairs using a strict center-distance threshold."""
    threshold = positive_scalar(collision_distance_m, name="collision_distance_m")
    distances = pairwise_distances(positions)
    num_agents = distances.shape[0]
    if num_agents == 1:
        collided = np.zeros(1, dtype=np.bool_)
        collided.setflags(write=False)
        return CollisionStatistics(0, collided, 0.0)

    upper_rows, upper_columns = np.triu_indices(num_agents, k=1)
    pair_distances = distances[upper_rows, upper_columns]
    collision_entries = pair_distances < threshold
    collided = np.zeros(num_agents, dtype=np.bool_)
    collided[upper_rows[collision_entries]] = True
    collided[upper_columns[collision_entries]] = True
    collided.setflags(write=False)
    return CollisionStatistics(
        pair_count=int(np.count_nonzero(collision_entries)),
        collided_agents=collided,
        minimum_separation_m=float(pair_distances.min()),
    )


def evaluate_kinematic_state(
    positions: ArrayLike,
    targets: ArrayLike,
    actions: ArrayLike,
    previous_actions: ArrayLike,
    *,
    collision_distance_m: float,
    success_tolerance_m: float,
) -> KinematicStateMetrics:
    """Evaluate a transition without depending on its training reward."""
    position_array, target_array = matching_point_arrays(positions, targets)
    action_array, previous_action_array = matching_point_arrays(actions, previous_actions)
    if action_array.shape[0] != position_array.shape[0]:
        raise ValueError("actions must contain one row per agent.")
    tolerance = positive_scalar(success_tolerance_m, name="success_tolerance_m")
    collision = collision_statistics(
        position_array,
        collision_distance_m=collision_distance_m,
    )
    assigned_position_rmse = position_rmse(position_array, target_array)
    action_delta = action_array - previous_action_array
    return KinematicStateMetrics(
        position_rmse_m=assigned_position_rmse,
        shape_rmse_m=shape_rmse(position_array, target_array),
        normalized_shape_rmse=normalized_shape_rmse(position_array, target_array),
        centroid_error_m=float(
            np.linalg.norm(position_array.mean(axis=0) - target_array.mean(axis=0))
        ),
        minimum_separation_m=collision.minimum_separation_m,
        collision_pairs=collision.pair_count,
        control_delta_rms=math.sqrt(float(np.mean(action_delta**2))),
        within_tolerance=assigned_position_rmse <= tolerance and collision.pair_count == 0,
    )
