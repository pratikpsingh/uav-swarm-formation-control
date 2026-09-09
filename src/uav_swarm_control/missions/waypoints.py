"""Deterministic waypoint construction and progress tracking."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray


def _point(value: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite 3D point.")
    return result


def linear_waypoints(start: ArrayLike, goal: ArrayLike, spacing_m: float) -> FloatArray:
    """Include a final goal and insert equally spaced points along a 3D segment."""
    first = _point(start, name="start")
    last = _point(goal, name="goal")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive.")
    distance = float(np.linalg.norm(last - first))
    if distance == 0.0:
        return last.reshape(1, 3).copy()
    segments = max(1, int(np.ceil(distance / spacing_m)))
    fractions = np.linspace(1.0 / segments, 1.0, segments)
    return first + fractions[:, None] * (last - first)


@dataclass(slots=True)
class WaypointTracker:
    """Advance only after the centroid stays within tolerance for a fixed hold."""

    waypoints: FloatArray
    tolerance_m: float
    hold_steps: int
    index: int = 0
    streak: int = 0

    def __post_init__(self) -> None:
        values = np.asarray(self.waypoints, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 1:
            raise ValueError("waypoints must have shape [count, 3] with count >= 1.")
        if not np.isfinite(values).all() or self.tolerance_m <= 0.0 or self.hold_steps < 1:
            raise ValueError("waypoints and tracker thresholds must be valid and positive.")
        self.waypoints = values.copy()

    @property
    def current(self) -> FloatArray:
        return self.waypoints[self.index].copy()

    @property
    def complete(self) -> bool:
        return self.index == len(self.waypoints) - 1 and self.streak >= self.hold_steps

    def update(self, centroid: ArrayLike) -> bool:
        """Update the hold streak and return whether the active waypoint changed."""
        point = _point(centroid, name="centroid")
        self.streak = (
            self.streak + 1 if np.linalg.norm(point - self.current) <= self.tolerance_m else 0
        )
        if self.streak < self.hold_steps or self.index == len(self.waypoints) - 1:
            return False
        self.index += 1
        self.streak = 0
        return True
