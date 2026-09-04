"""Validated observations shared by environments and learning algorithms."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uav_swarm_control._arrays import (
    BoolArray,
    Float32Array,
    immutable_bool_array,
    immutable_float32_array,
    validated_agent_ids,
)
from uav_swarm_control.agents import AgentId


@dataclass(frozen=True, slots=True)
class LocalObservations:
    """Batched local information available to decentralized actors.

    Ego values have shape (N, E), neighbors have shape (N, K, F), and
    the neighbor mask has shape (N, K). False mask entries are padding
    and their feature rows must contain zeros.
    """

    agent_ids: Sequence[AgentId]
    ego: Float32Array
    neighbors: Float32Array
    neighbor_mask: BoolArray

    def __post_init__(self) -> None:
        ego = immutable_float32_array(self.ego, name="ego", dimensions=2)
        neighbors = immutable_float32_array(self.neighbors, name="neighbors", dimensions=3)
        mask = immutable_bool_array(self.neighbor_mask, name="neighbor_mask", dimensions=2)

        num_agents = ego.shape[0]
        if num_agents < 1:
            raise ValueError("ego must contain at least one agent.")
        if ego.shape[1] < 1:
            raise ValueError("ego must contain at least one feature.")
        if neighbors.shape[0] != num_agents:
            raise ValueError("ego and neighbors must have the same agent dimension.")
        if neighbors.shape[2] < 1:
            raise ValueError("neighbors must contain at least one feature per slot.")
        if mask.shape != neighbors.shape[:2]:
            raise ValueError(
                "neighbor_mask must match the agent and neighbor dimensions; "
                f"received {mask.shape} for neighbors {neighbors.shape}."
            )
        if np.any(neighbors[~mask] != 0.0):
            raise ValueError("masked neighbor slots must contain only zero padding.")

        identifiers = validated_agent_ids(self.agent_ids, expected=num_agents)
        object.__setattr__(self, "agent_ids", identifiers)
        object.__setattr__(self, "ego", ego)
        object.__setattr__(self, "neighbors", neighbors)
        object.__setattr__(self, "neighbor_mask", mask)

    @property
    def num_agents(self) -> int:
        """Number of agents represented in the batch."""
        return self.ego.shape[0]

    @property
    def max_neighbors(self) -> int:
        """Number of padded neighbor slots per agent."""
        return self.neighbors.shape[1]


@dataclass(frozen=True, slots=True)
class CentralizedState:
    """Training-only privileged state for a centralized critic."""

    values: Float32Array

    def __post_init__(self) -> None:
        values = immutable_float32_array(self.values, name="centralized state", dimensions=1)
        if values.size < 1:
            raise ValueError("centralized state must contain at least one feature.")
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        """Number of centralized state features."""
        return self.values.size
