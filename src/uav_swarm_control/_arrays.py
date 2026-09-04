"""Array validation shared by model-facing contracts."""

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_swarm_control.agents import AgentId

type Float32Array = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]


def immutable_float32_array(
    values: ArrayLike,
    *,
    name: str,
    dimensions: int,
) -> Float32Array:
    """Return a finite, immutable float32 copy with the requested rank."""
    try:
        array = np.array(values, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric.") from error
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions; received {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def immutable_bool_array(values: ArrayLike, *, name: str, dimensions: int) -> BoolArray:
    """Return an immutable copy of a strictly boolean array."""
    raw = np.asarray(values)
    if raw.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must use the bool dtype.")
    array = np.array(raw, dtype=np.bool_, copy=True)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions; received {array.shape}.")
    array.setflags(write=False)
    return array


def validated_agent_ids(values: Sequence[object], *, expected: int) -> tuple[AgentId, ...]:
    """Return unique agent identifiers with the expected length."""
    identifiers = tuple(values)
    if len(identifiers) != expected:
        raise ValueError(
            f"agent_ids must contain {expected} identifiers; received {len(identifiers)}."
        )
    if any(not isinstance(identifier, AgentId) for identifier in identifiers):
        raise TypeError("agent_ids must contain only AgentId values.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("agent_ids must be unique.")
    return cast(tuple[AgentId, ...], identifiers)
