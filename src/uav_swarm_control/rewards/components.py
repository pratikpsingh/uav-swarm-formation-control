"""Individually inspectable reward components for kinematic control."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control._arrays import Float32Array, immutable_float32_array
from uav_swarm_control.configuration.models import RewardConfig
from uav_swarm_control.evaluation.kinematics import collision_statistics
from uav_swarm_control.formations import normalized_shape_mse, squared_position_errors
from uav_swarm_control.formations._validation import matching_point_arrays


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """One vector for every reward term and their elementwise total."""

    navigation: Float32Array
    formation: Float32Array
    collision: Float32Array
    smoothness: Float32Array
    termination: Float32Array

    def __post_init__(self) -> None:
        arrays = {
            name: immutable_float32_array(values, name=name, dimensions=1)
            for name, values in (
                ("navigation reward", self.navigation),
                ("formation reward", self.formation),
                ("collision reward", self.collision),
                ("smoothness reward", self.smoothness),
                ("termination reward", self.termination),
            )
        }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("all reward components must have the same shape.")
        for field_name, array in zip(
            ("navigation", "formation", "collision", "smoothness", "termination"),
            arrays.values(),
            strict=True,
        ):
            object.__setattr__(self, field_name, array)

    @property
    def total(self) -> Float32Array:
        """Return the elementwise sum of all components."""
        return immutable_float32_array(
            self.navigation + self.formation + self.collision + self.smoothness + self.termination,
            name="total reward",
            dimensions=1,
        )

    def mean_metrics(self) -> dict[str, float]:
        """Return component means for transition diagnostics."""
        return {
            "reward/navigation_mean": float(self.navigation.mean()),
            "reward/formation_mean": float(self.formation.mean()),
            "reward/collision_mean": float(self.collision.mean()),
            "reward/smoothness_mean": float(self.smoothness.mean()),
            "reward/termination_mean": float(self.termination.mean()),
            "reward/total_mean": float(self.total.mean()),
        }


def compute_reward(
    positions: ArrayLike,
    targets: ArrayLike,
    actions: ArrayLike,
    previous_actions: ArrayLike,
    *,
    collision_distance_m: float,
    success: bool,
    config: RewardConfig,
) -> RewardBreakdown:
    """Compute transparent per-agent shaping and terminal reward terms."""
    position_array, target_array = matching_point_arrays(positions, targets)
    action_array, previous_action_array = matching_point_arrays(actions, previous_actions)
    if action_array.shape[0] != position_array.shape[0]:
        raise ValueError("actions must contain one row per agent.")

    num_agents = position_array.shape[0]
    distances = np.sqrt(squared_position_errors(position_array, target_array))
    shape_cost = normalized_shape_mse(position_array, target_array)
    collision = collision_statistics(
        position_array,
        collision_distance_m=collision_distance_m,
    )
    action_delta_cost = np.mean((action_array - previous_action_array) ** 2, axis=1)
    success_values = np.full(num_agents, float(success), dtype=np.float64)

    return RewardBreakdown(
        navigation=np.asarray(-config.navigation_weight * distances, dtype=np.float32),
        formation=np.full(
            num_agents,
            -config.formation_weight * shape_cost,
            dtype=np.float32,
        ),
        collision=np.asarray(
            -config.collision_penalty * collision.collided_agents,
            dtype=np.float32,
        ),
        smoothness=np.asarray(
            -config.smoothness_weight * action_delta_cost,
            dtype=np.float32,
        ),
        termination=np.asarray(config.success_bonus * success_values, dtype=np.float32),
    )
