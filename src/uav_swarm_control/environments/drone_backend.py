"""Typed boundary between project environments and rigid-body simulators."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from uav_swarm_control._arrays import Float32Array, immutable_float32_array

type SimulatorMetadataValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class RigidBodyState:
    """Canonical multi-drone state returned by a physics backend."""

    positions: Float32Array
    quaternions_xyzw: Float32Array
    euler_angles_radians: Float32Array
    linear_velocities_mps: Float32Array
    angular_velocities_rad_s: Float32Array
    motor_rpms: Float32Array

    def __post_init__(self) -> None:
        fields = {
            "positions": (self.positions, 3),
            "quaternions_xyzw": (self.quaternions_xyzw, 4),
            "euler_angles_radians": (self.euler_angles_radians, 3),
            "linear_velocities_mps": (self.linear_velocities_mps, 3),
            "angular_velocities_rad_s": (self.angular_velocities_rad_s, 3),
            "motor_rpms": (self.motor_rpms, 4),
        }
        canonical: dict[str, Float32Array] = {}
        for name, (values, width) in fields.items():
            array = immutable_float32_array(values, name=name, dimensions=2)
            if array.shape[1] != width:
                raise ValueError(f"{name} must have shape (N, {width}); received {array.shape}.")
            canonical[name] = array
        row_counts = {array.shape[0] for array in canonical.values()}
        if len(row_counts) != 1 or next(iter(row_counts)) < 1:
            raise ValueError("rigid-body state fields must contain the same positive row count.")
        for name, array in canonical.items():
            object.__setattr__(self, name, array)

    @property
    def num_drones(self) -> int:
        """Number of drones represented in every state field."""
        return self.positions.shape[0]


def immutable_simulator_metadata(
    values: Mapping[str, SimulatorMetadataValue],
) -> Mapping[str, SimulatorMetadataValue]:
    """Copy and freeze scalar provenance values."""
    result: dict[str, SimulatorMetadataValue] = {}
    for name, typed_value in values.items():
        value = cast(object, typed_value)
        if not name or name.strip() != name:
            raise ValueError("simulator metadata names must be non-empty and trimmed.")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"simulator metadata {name!r} must be a scalar value.")
        result[name] = value
    return MappingProxyType(result)


@runtime_checkable
class DronePhysicsBackend(Protocol):
    """Minimal simulator surface required by the common swarm environment."""

    @property
    def metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        """Return immutable simulator and timing provenance."""
        ...

    def reset(self, initial_positions: Float32Array, *, seed: int) -> RigidBodyState:
        """Reset all drones and controller memory to a known episode state."""
        ...

    def step_velocity(self, target_velocities_mps: Float32Array) -> RigidBodyState:
        """Apply one velocity-to-motor-control interval and advance physics."""
        ...

    def close(self) -> None:
        """Release simulator resources; repeated calls must be safe."""
        ...
