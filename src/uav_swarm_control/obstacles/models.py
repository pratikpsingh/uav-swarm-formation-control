"""Simulator-independent kinematic spherical-obstacle models."""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations._validation import matching_point_arrays, points_array

type BoolArray = NDArray[np.bool_]


class ObstacleScenario(StrEnum):
    """Controlled obstacle distributions used for training and evaluation."""

    NONE = "no-obstacle"
    STATIC = "static"
    SLOW_DYNAMIC = "slow-dynamic"
    MIXED_DYNAMIC = "mixed-dynamic"


@dataclass(frozen=True, slots=True)
class ObstacleFieldConfig:
    """Geometry, sensing, reward, and sampling bounds for spherical obstacles."""

    max_obstacles: int
    static_count: int
    slow_dynamic_count: int
    mixed_count: int
    vehicle_radius_m: float
    radius_lower_m: float
    radius_upper_m: float
    speed_lower_mps: float
    speed_upper_mps: float
    route_fraction_lower: float
    route_fraction_upper: float
    static_lateral_lower_m: float
    static_lateral_upper_m: float
    dynamic_lateral_lower_m: float
    dynamic_lateral_upper_m: float
    vertical_offset_lower_m: float
    vertical_offset_upper_m: float
    mixed_dynamic_probability: float
    observation_radius_m: float | None
    safety_margin_m: float
    proximity_weight: float
    collision_penalty: float

    def __post_init__(self) -> None:
        counts = (self.max_obstacles, self.static_count, self.slow_dynamic_count, self.mixed_count)
        if any(isinstance(value, bool) for value in counts):
            raise ValueError("obstacle counts must be integers.")
        if self.max_obstacles < 1 or any(
            value < 1 or value > self.max_obstacles for value in counts[1:]
        ):
            raise ValueError("scenario counts must lie between one and max_obstacles.")
        positive = (
            self.vehicle_radius_m,
            self.radius_lower_m,
            self.radius_upper_m,
            self.speed_lower_mps,
            self.speed_upper_mps,
            self.safety_margin_m,
            self.collision_penalty,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("radii, speeds, margin, and collision penalty must be positive.")
        if self.radius_lower_m > self.radius_upper_m or self.speed_lower_mps > self.speed_upper_mps:
            raise ValueError("obstacle radius and speed lower bounds cannot exceed upper bounds.")
        ranges = (
            (self.route_fraction_lower, self.route_fraction_upper),
            (self.static_lateral_lower_m, self.static_lateral_upper_m),
            (self.dynamic_lateral_lower_m, self.dynamic_lateral_upper_m),
            (self.vertical_offset_lower_m, self.vertical_offset_upper_m),
        )
        if any(
            not math.isfinite(lower) or not math.isfinite(upper) or lower > upper
            for lower, upper in ranges
        ):
            raise ValueError("obstacle sampling ranges require finite ordered bounds.")
        if not 0.0 < self.route_fraction_lower <= self.route_fraction_upper < 1.0:
            raise ValueError("route fractions must lie strictly between zero and one.")
        if self.static_lateral_lower_m < 0.0 or self.dynamic_lateral_lower_m <= 0.0:
            raise ValueError("lateral-distance ranges cannot be negative or dynamic zero.")
        if not 0.0 <= self.mixed_dynamic_probability <= 1.0:
            raise ValueError("mixed_dynamic_probability must lie in [0, 1].")
        if self.observation_radius_m is not None and (
            not math.isfinite(self.observation_radius_m) or self.observation_radius_m <= 0.0
        ):
            raise ValueError("observation_radius_m must be null or positive.")
        if not math.isfinite(self.proximity_weight) or self.proximity_weight < 0.0:
            raise ValueError("proximity_weight must be finite and non-negative.")

    def count_for(self, scenario: ObstacleScenario) -> int:
        """Return the configured number of obstacles for one scenario."""
        return {
            ObstacleScenario.NONE: 0,
            ObstacleScenario.STATIC: self.static_count,
            ObstacleScenario.SLOW_DYNAMIC: self.slow_dynamic_count,
            ObstacleScenario.MIXED_DYNAMIC: self.mixed_count,
        }[scenario]


@dataclass(frozen=True, slots=True)
class ObstacleField:
    """Current obstacle centers, constant velocities, and sphere radii."""

    positions_m: FloatArray
    velocities_mps: FloatArray
    radii_m: FloatArray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64).copy()
        velocities = np.asarray(self.velocities_mps, dtype=np.float64).copy()
        radii = np.asarray(self.radii_m, dtype=np.float64).copy()
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("obstacle positions must have shape (M, 3).")
        if velocities.shape != positions.shape or radii.shape != (len(positions),):
            raise ValueError("obstacle velocities/radii must match the position rows.")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("obstacle states must be finite.")
        if not np.isfinite(radii).all() or np.any(radii <= 0.0):
            raise ValueError("obstacle radii must be finite and positive.")
        for array in (positions, velocities, radii):
            array.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "radii_m", radii)

    @property
    def count(self) -> int:
        return len(self.radii_m)

    @property
    def dynamic_count(self) -> int:
        return int(np.count_nonzero(np.linalg.norm(self.velocities_mps, axis=1) > 0.0))


@dataclass(frozen=True, slots=True)
class ObstacleSafety:
    """Surface clearances and collisions between UAV spheres and obstacles."""

    clearances_m: FloatArray
    collided_agents: BoolArray
    collision_pairs: int
    minimum_clearance_m: float | None


def empty_obstacle_field() -> ObstacleField:
    """Return a valid field with no active obstacles."""
    return ObstacleField(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))


def advance_obstacles(field: ObstacleField, time_step_seconds: float) -> ObstacleField:
    """Advance constant-velocity obstacle centers by one exact time step."""
    if not math.isfinite(time_step_seconds) or time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive.")
    return ObstacleField(
        field.positions_m + field.velocities_mps * time_step_seconds,
        field.velocities_mps,
        field.radii_m,
    )


def obstacle_safety(
    agent_positions_m: ArrayLike,
    field: ObstacleField,
    *,
    vehicle_radius_m: float,
) -> ObstacleSafety:
    """Measure sampled sphere-surface clearance without simulator dependencies."""
    agents = points_array(agent_positions_m, name="agent_positions_m")
    if not math.isfinite(vehicle_radius_m) or vehicle_radius_m <= 0.0:
        raise ValueError("vehicle_radius_m must be positive.")
    clearances = np.linalg.norm(agents[:, None, :] - field.positions_m[None, :, :], axis=2) - (
        vehicle_radius_m + field.radii_m[None, :]
    )
    collisions = clearances < 0.0
    collided_agents = np.any(collisions, axis=1)
    clearances.setflags(write=False)
    collided_agents.setflags(write=False)
    return ObstacleSafety(
        clearances,
        collided_agents,
        int(np.count_nonzero(collisions)),
        None if field.count == 0 else float(clearances.min()),
    )


def _route_basis(
    origins: FloatArray, targets: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    start = origins.mean(axis=0)
    displacement = targets.mean(axis=0) - start
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-9:
        raise ValueError("obstacle sampling requires separated initial and target centroids.")
    forward = displacement / distance
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(forward, reference)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(lateral, forward)
    return displacement, lateral, vertical


def sample_obstacle_field(
    scenario: ObstacleScenario,
    origins_m: ArrayLike,
    targets_m: ArrayLike,
    config: ObstacleFieldConfig,
    rng: np.random.Generator,
) -> ObstacleField:
    """Sample feasible static or crossing spheres around the centroid route."""
    origins, targets = matching_point_arrays(origins_m, targets_m)
    scenario = ObstacleScenario(scenario)
    count = config.count_for(scenario)
    if count == 0:
        return empty_obstacle_field()
    displacement, lateral, vertical = _route_basis(origins, targets)
    start = origins.mean(axis=0)
    for _field_attempt in range(64):
        positions: list[FloatArray] = []
        velocities: list[FloatArray] = []
        radii: list[float] = []
        for _ in range(count):
            for _obstacle_attempt in range(256):
                radius = float(rng.uniform(config.radius_lower_m, config.radius_upper_m))
                dynamic = scenario is ObstacleScenario.SLOW_DYNAMIC or (
                    scenario is ObstacleScenario.MIXED_DYNAMIC
                    and rng.random() < config.mixed_dynamic_probability
                )
                fraction = float(
                    rng.uniform(config.route_fraction_lower, config.route_fraction_upper)
                )
                sign = -1.0 if rng.random() < 0.5 else 1.0
                vertical_offset = float(
                    rng.uniform(config.vertical_offset_lower_m, config.vertical_offset_upper_m)
                )
                if dynamic:
                    lateral_distance = float(
                        rng.uniform(config.dynamic_lateral_lower_m, config.dynamic_lateral_upper_m)
                    )
                    speed = float(rng.uniform(config.speed_lower_mps, config.speed_upper_mps))
                    velocity = -sign * speed * lateral
                else:
                    lateral_distance = float(
                        rng.uniform(config.static_lateral_lower_m, config.static_lateral_upper_m)
                    )
                    velocity = np.zeros(3)
                candidate = (
                    start
                    + fraction * displacement
                    + sign * lateral_distance * lateral
                    + vertical_offset * vertical
                )
                initial_clearance = np.linalg.norm(origins - candidate, axis=1) - (
                    config.vehicle_radius_m + radius
                )
                target_clearance = np.linalg.norm(targets - candidate, axis=1) - (
                    config.vehicle_radius_m + radius
                )
                other_clearance = [
                    float(np.linalg.norm(candidate - other)) - radius - other_radius
                    for other, other_radius in zip(positions, radii, strict=True)
                ]
                if (
                    float(initial_clearance.min()) >= config.safety_margin_m
                    and float(target_clearance.min()) >= config.safety_margin_m
                    and all(clearance >= 0.05 for clearance in other_clearance)
                    and candidate[2] - radius > 0.0
                ):
                    positions.append(candidate)
                    velocities.append(velocity)
                    radii.append(radius)
                    break
            else:
                break
        else:
            return ObstacleField(np.stack(positions), np.stack(velocities), np.asarray(radii))
    raise ValueError("could not sample a feasible obstacle field from configured bounds.")


__all__ = [
    "ObstacleField",
    "ObstacleFieldConfig",
    "ObstacleSafety",
    "ObstacleScenario",
    "advance_obstacles",
    "empty_obstacle_field",
    "obstacle_safety",
    "sample_obstacle_field",
]
