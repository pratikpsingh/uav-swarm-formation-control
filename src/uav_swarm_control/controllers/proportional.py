"""Deterministic proportional position controller."""

import math
from dataclasses import dataclass

import numpy as np

from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.observations import LocalObservations


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class ProportionalPositionController:
    """Drive each agent toward its assigned target using local position error."""

    gain_per_second: float
    max_velocity_component_mps: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gain_per_second",
            _positive_finite(self.gain_per_second, name="gain_per_second"),
        )
        object.__setattr__(
            self,
            "max_velocity_component_mps",
            _positive_finite(
                self.max_velocity_component_mps,
                name="max_velocity_component_mps",
            ),
        )

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        """Apply velocity = gain times assigned-target displacement."""
        if observations.ego.shape[1] < 3:
            raise ValueError("ego observations must begin with three relative-target features.")
        relative_targets = observations.ego[:, :3]
        desired_velocity = self.gain_per_second * relative_targets
        normalized = np.clip(
            desired_velocity / self.max_velocity_component_mps,
            -1.0,
            1.0,
        ).astype(np.float32)
        return NormalizedVelocityActions(observations.agent_ids, normalized)
