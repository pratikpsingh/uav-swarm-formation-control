"""Threshold-based recovery measurements for disturbance experiments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    peak_error: float
    error_area: float
    recovery_seconds: float | None
    final_error: float


def measure_recovery(
    errors: ArrayLike,
    *,
    threshold: float,
    hold_steps: int,
    time_step_seconds: float,
) -> RecoveryMetrics:
    """Measure post-perturbation error and sustained return below a fixed threshold."""
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("errors must be a non-empty finite one-dimensional series.")
    if threshold < 0.0 or hold_steps < 1 or time_step_seconds <= 0.0:
        raise ValueError("recovery threshold, hold, and time step are invalid.")
    recovered: int | None = None
    for start in range(0, len(values) - hold_steps + 1):
        if np.all(values[start : start + hold_steps] <= threshold):
            recovered = start + hold_steps - 1
            break
    return RecoveryMetrics(
        peak_error=float(values.max()),
        error_area=float(np.trapezoid(values, dx=time_step_seconds)),
        recovery_seconds=None if recovered is None else recovered * time_step_seconds,
        final_error=float(values[-1]),
    )
