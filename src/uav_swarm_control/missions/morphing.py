"""Rate-limited formation deformation and restoration logic."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray


class MorphingPhase(StrEnum):
    NOMINAL = "nominal"
    DEFORMING = "deforming"
    AVOIDING = "avoiding"
    RESTORING = "restoring"


def _targets(value: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 1:
        raise ValueError(f"{name} must have shape [agents, 3].")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values.")
    return result


def interpolate_targets(nominal: ArrayLike, avoidance: ArrayLike, alpha: float) -> FloatArray:
    """Interpolate corresponding UAV targets without changing target identity."""
    first = _targets(nominal, name="nominal targets")
    second = _targets(avoidance, name="avoidance targets")
    if first.shape != second.shape:
        raise ValueError("nominal and avoidance targets must have identical shapes.")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    return (1.0 - alpha) * first + alpha * second


@dataclass(slots=True)
class MorphingController:
    """High-level deterministic planner; the learned actor tracks active targets."""

    alpha_rate_per_second: float
    release_hold_steps: int
    alpha: float = 0.0
    phase: MorphingPhase = MorphingPhase.NOMINAL
    release_streak: int = 0
    deformation_steps: int = 0
    avoidance_steps: int = 0
    restoration_steps: int = 0

    def __post_init__(self) -> None:
        if self.alpha_rate_per_second <= 0.0 or self.release_hold_steps < 1:
            raise ValueError("morph rate and release hold must be positive.")

    def update(self, *, blocked: bool, time_step_seconds: float) -> MorphingPhase:
        """Advance the morph phase and bounded interpolation factor."""
        if time_step_seconds <= 0.0:
            raise ValueError("time_step_seconds must be positive.")
        increment = min(1.0, self.alpha_rate_per_second * time_step_seconds)
        if self.phase is MorphingPhase.NOMINAL and blocked:
            self.phase = MorphingPhase.DEFORMING
        if self.phase is MorphingPhase.DEFORMING:
            self.deformation_steps += 1
            self.alpha = min(1.0, self.alpha + increment)
            if self.alpha == 1.0:
                self.phase = MorphingPhase.AVOIDING
        elif self.phase is MorphingPhase.AVOIDING:
            self.avoidance_steps += 1
            self.release_streak = 0 if blocked else self.release_streak + 1
            if self.release_streak >= self.release_hold_steps:
                self.phase = MorphingPhase.RESTORING
        elif self.phase is MorphingPhase.RESTORING:
            self.restoration_steps += 1
            if blocked:
                self.phase = MorphingPhase.DEFORMING
            else:
                self.alpha = max(0.0, self.alpha - increment)
                if self.alpha == 0.0:
                    self.phase = MorphingPhase.NOMINAL
                    self.release_streak = 0
        return self.phase
