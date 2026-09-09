"""Simulator-independent definitions for controlled recovery perturbations."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray


class DisturbanceKind(StrEnum):
    POSITION = "position"
    VELOCITY = "velocity"


@dataclass(frozen=True, slots=True)
class Disturbance:
    """A predeclared perturbation applied to selected agent rows at one step."""

    kind: DisturbanceKind
    step: int
    agent_indices: tuple[int, ...]
    vector: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.step < 1 or not self.agent_indices or min(self.agent_indices) < 0:
            raise ValueError("disturbance step and selected agent indices must be valid.")
        if len(set(self.agent_indices)) != len(self.agent_indices):
            raise ValueError("disturbance agent indices must be unique.")
        if len(self.vector) != 3 or not np.isfinite(self.vector).all():
            raise ValueError("disturbance vector must contain three finite values.")
        object.__setattr__(self, "kind", DisturbanceKind(self.kind))


def apply_disturbance(values: ArrayLike, disturbance: Disturbance) -> FloatArray:
    """Return a perturbed position or velocity matrix without mutating its input."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("state values must have shape [agents, 3].")
    if max(disturbance.agent_indices) >= len(result):
        raise ValueError("disturbance selects an unavailable agent row.")
    result[list(disturbance.agent_indices)] += np.asarray(disturbance.vector)
    return result


__all__ = ["Disturbance", "DisturbanceKind", "apply_disturbance"]
