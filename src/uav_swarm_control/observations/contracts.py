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

    Ego values have shape (N, E), neighbors have shape (N, K, F), obstacles
    have shape (N, M, 7), and each variable-cardinality axis has a mask.
    False mask entries are padding and their feature rows must contain zeros.
    """

    agent_ids: Sequence[AgentId]
    ego: Float32Array
    neighbors: Float32Array
    neighbor_mask: BoolArray
    obstacles: Float32Array | None = None
    obstacle_mask: BoolArray | None = None

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

        if (self.obstacles is None) != (self.obstacle_mask is None):
            raise ValueError("obstacles and obstacle_mask must either both be provided or omitted.")
        obstacles = immutable_float32_array(
            np.zeros((num_agents, 0, 7), dtype=np.float32)
            if self.obstacles is None
            else self.obstacles,
            name="obstacles",
            dimensions=3,
        )
        obstacle_mask = immutable_bool_array(
            np.zeros((num_agents, 0), dtype=np.bool_)
            if self.obstacle_mask is None
            else self.obstacle_mask,
            name="obstacle_mask",
            dimensions=2,
        )
        if obstacles.shape[0] != num_agents or obstacles.shape[2] != 7:
            raise ValueError("obstacles must have shape (N, M, 7).")
        if obstacle_mask.shape != obstacles.shape[:2]:
            raise ValueError("obstacle_mask must match the obstacle agent and slot dimensions.")
        if np.any(obstacles[~obstacle_mask] != 0.0):
            raise ValueError("masked obstacle slots must contain only zero padding.")

        identifiers = validated_agent_ids(self.agent_ids, expected=num_agents)
        object.__setattr__(self, "agent_ids", identifiers)
        object.__setattr__(self, "ego", ego)
        object.__setattr__(self, "neighbors", neighbors)
        object.__setattr__(self, "neighbor_mask", mask)
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "obstacle_mask", obstacle_mask)

    @property
    def num_agents(self) -> int:
        """Number of agents represented in the batch."""
        return self.ego.shape[0]

    @property
    def max_neighbors(self) -> int:
        """Number of padded neighbor slots per agent."""
        return self.neighbors.shape[1]

    @property
    def max_obstacles(self) -> int:
        """Number of padded spherical-obstacle slots per agent."""
        assert self.obstacles is not None
        return self.obstacles.shape[1]


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
