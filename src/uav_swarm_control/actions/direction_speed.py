"""Paper-aligned direction-plus-speed action conversion."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uav_swarm_control._arrays import Float32Array, immutable_float32_array, validated_agent_ids
from uav_swarm_control.agents import AgentId
from uav_swarm_control.environments.contracts import NormalizedVelocityActions

DIRECTION_SPEED_ACTION_SIZE = 4


@dataclass(frozen=True, slots=True)
class DirectionSpeedActions:
    """Bounded `[direction_x, direction_y, direction_z, speed]` actor outputs."""

    agent_ids: Sequence[AgentId]
    values: Float32Array

    def __post_init__(self) -> None:
        values = immutable_float32_array(self.values, name="direction-speed actions", dimensions=2)
        if values.shape[0] < 1 or values.shape[1] != DIRECTION_SPEED_ACTION_SIZE:
            raise ValueError(
                "direction-speed actions must have shape (N, 4) with N >= 1; "
                f"received {values.shape}."
            )
        if np.any(np.abs(values) > 1.0):
            raise ValueError("direction-speed actions must lie in [-1, 1].")
        object.__setattr__(
            self,
            "agent_ids",
            validated_agent_ids(self.agent_ids, expected=values.shape[0]),
        )
        object.__setattr__(self, "values", values)


def direction_speed_to_normalized_velocity(
    actions: DirectionSpeedActions,
    *,
    direction_epsilon: float = 1e-6,
) -> NormalizedVelocityActions:
    """Convert a bounded direction and speed fraction into normalized 3D velocity.

    The fourth component is mapped from ``[-1, 1]`` to ``[0, 1]``. Direction is
    normalized, so the environment's configured maximum speed remains the only
    physical scale. A direction whose norm is below ``direction_epsilon`` maps to
    zero velocity, irrespective of the speed component.
    """
    if not np.isfinite(direction_epsilon) or direction_epsilon <= 0.0:
        raise ValueError("direction_epsilon must be finite and positive.")
    values = np.asarray(actions.values, dtype=np.float64)
    directions = values[:, :3]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = np.divide(
        directions,
        norms,
        out=np.zeros_like(directions),
        where=norms >= direction_epsilon,
    )
    speed_fractions = (values[:, 3:4] + 1.0) / 2.0
    normalized_velocity = unit_directions * speed_fractions
    return NormalizedVelocityActions(
        actions.agent_ids,
        normalized_velocity.astype(np.float32),
    )
