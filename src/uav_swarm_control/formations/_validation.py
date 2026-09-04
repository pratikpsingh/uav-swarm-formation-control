"""Runtime validation shared by formation modules."""

import math

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray


def points_array(values: ArrayLike, *, name: str, minimum_points: int = 1) -> FloatArray:
    """Return a finite float64 point array with shape (N, 3)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); received {array.shape}.")
    if array.shape[0] < minimum_points:
        raise ValueError(f"{name} must contain at least {minimum_points} point(s).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def vector3(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite float64 vector with shape (3,)."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,); received {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def finite_scalar(value: object, *, name: str) -> float:
    """Validate and return a finite scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def positive_scalar(value: float, *, name: str) -> float:
    """Validate and return a finite scalar greater than zero."""
    result = finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def agent_count(
    value: object,
    *,
    minimum: int = 1,
    exact: int | None = None,
) -> int:
    """Validate an agent count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("num_agents must be an integer.")
    if exact is not None and value != exact:
        raise ValueError(f"num_agents must be {exact} for this formation; received {value}.")
    if value < minimum:
        raise ValueError(f"num_agents must be at least {minimum}; received {value}.")
    return value


def matching_point_arrays(current: ArrayLike, target: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate two point arrays with fixed row correspondence."""
    current_array = points_array(current, name="current")
    target_array = points_array(target, name="target")
    if current_array.shape != target_array.shape:
        raise ValueError(
            "current and target must have identical shapes; "
            f"received {current_array.shape} and {target_array.shape}."
        )
    return current_array, target_array
