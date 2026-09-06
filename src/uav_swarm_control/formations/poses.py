"""Simulator-independent target-pose ranges and deterministic sampling."""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations.transforms import rotation_matrix_from_euler

type Vector3 = tuple[float, float, float]


def _finite_vector(values: Vector3, *, name: str) -> Vector3:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite values.")
    return result


@dataclass(frozen=True, slots=True)
class FormationPoseRange:
    """Closed uniform ranges applied around the configured nominal target pose."""

    center_offset_lower_m: Vector3
    center_offset_upper_m: Vector3
    euler_offset_lower_radians: Vector3
    euler_offset_upper_radians: Vector3
    scale_lower: float
    scale_upper: float

    def __post_init__(self) -> None:
        lower_center = _finite_vector(self.center_offset_lower_m, name="center offset lower")
        upper_center = _finite_vector(self.center_offset_upper_m, name="center offset upper")
        lower_euler = _finite_vector(self.euler_offset_lower_radians, name="Euler offset lower")
        upper_euler = _finite_vector(self.euler_offset_upper_radians, name="Euler offset upper")
        lower_scale = float(self.scale_lower)
        upper_scale = float(self.scale_upper)
        if not all(math.isfinite(value) for value in (lower_scale, upper_scale)):
            raise ValueError("pose scales must be finite.")
        if lower_scale <= 0.0 or lower_scale > upper_scale:
            raise ValueError("pose scale range must satisfy 0 < lower <= upper.")
        if any(lower > upper for lower, upper in zip(lower_center, upper_center, strict=True)):
            raise ValueError("center-offset lower bounds cannot exceed upper bounds.")
        if any(lower > upper for lower, upper in zip(lower_euler, upper_euler, strict=True)):
            raise ValueError("Euler-offset lower bounds cannot exceed upper bounds.")
        if any(abs(value) > math.pi for value in (*lower_euler, *upper_euler)):
            raise ValueError("Euler offsets must lie in [-pi, pi].")
        object.__setattr__(self, "center_offset_lower_m", lower_center)
        object.__setattr__(self, "center_offset_upper_m", upper_center)
        object.__setattr__(self, "euler_offset_lower_radians", lower_euler)
        object.__setattr__(self, "euler_offset_upper_radians", upper_euler)
        object.__setattr__(self, "scale_lower", lower_scale)
        object.__setattr__(self, "scale_upper", upper_scale)


@dataclass(frozen=True, slots=True)
class FormationPose:
    """One sampled world pose and the proper rotation derived from its Euler angles."""

    center_m: Vector3
    euler_radians: Vector3
    scale: float
    rotation: FloatArray


def pose_ranges_overlap(first: FormationPoseRange, second: FormationPoseRange) -> bool:
    """Return whether two seven-dimensional closed range boxes intersect."""
    first_lower = (
        *first.center_offset_lower_m,
        *first.euler_offset_lower_radians,
        first.scale_lower,
    )
    first_upper = (
        *first.center_offset_upper_m,
        *first.euler_offset_upper_radians,
        first.scale_upper,
    )
    second_lower = (
        *second.center_offset_lower_m,
        *second.euler_offset_lower_radians,
        second.scale_lower,
    )
    second_upper = (
        *second.center_offset_upper_m,
        *second.euler_offset_upper_radians,
        second.scale_upper,
    )
    return all(
        max(lower_a, lower_b) <= min(upper_a, upper_b)
        for lower_a, upper_a, lower_b, upper_b in zip(
            first_lower,
            first_upper,
            second_lower,
            second_upper,
            strict=True,
        )
    )


def sample_formation_pose(
    nominal_center_m: Vector3,
    nominal_euler_radians: Vector3,
    pose_range: FormationPoseRange,
    rng: np.random.Generator,
) -> FormationPose:
    """Sample one pose without coupling it to initial-state noise."""
    center_offset = rng.uniform(
        pose_range.center_offset_lower_m,
        pose_range.center_offset_upper_m,
    )
    euler_offset = rng.uniform(
        pose_range.euler_offset_lower_radians,
        pose_range.euler_offset_upper_radians,
    )
    center = tuple(
        float(base + offset) for base, offset in zip(nominal_center_m, center_offset, strict=True)
    )
    euler = tuple(
        float(base + offset)
        for base, offset in zip(nominal_euler_radians, euler_offset, strict=True)
    )
    scale = float(rng.uniform(pose_range.scale_lower, pose_range.scale_upper))
    return FormationPose(
        cast(Vector3, center),
        cast(Vector3, euler),
        scale,
        rotation_matrix_from_euler(*euler),
    )


__all__ = [
    "FormationPose",
    "FormationPoseRange",
    "pose_ranges_overlap",
    "sample_formation_pose",
]
