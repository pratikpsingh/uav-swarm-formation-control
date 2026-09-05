"""Simulator-independent builders for swarm observations and critic state."""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.agents import AgentId
from uav_swarm_control.formations._validation import matching_point_arrays
from uav_swarm_control.observations.contracts import CentralizedState, LocalObservations


def build_local_observations(
    agent_ids: Sequence[AgentId],
    positions: ArrayLike,
    velocities: ArrayLike,
    targets: ArrayLike,
    *,
    max_neighbors: int,
    neighbor_radius_m: float | None,
) -> LocalObservations:
    """Build the common ego, ordered-neighbor, and validity-mask representation."""
    position_array, target_array = matching_point_arrays(positions, targets)
    velocity_array, _ = matching_point_arrays(velocities, positions)
    num_agents = len(agent_ids)
    if position_array.shape[0] != num_agents:
        raise ValueError("positions must contain one row per agent identifier.")
    if max_neighbors < 0 or max_neighbors > num_agents - 1:
        raise ValueError("max_neighbors must be between zero and num_agents - 1.")

    relative_targets = target_array - position_array
    ego = np.concatenate((relative_targets, velocity_array), axis=1).astype(np.float32)
    neighbors = np.zeros((num_agents, max_neighbors, 6), dtype=np.float32)
    mask = np.zeros((num_agents, max_neighbors), dtype=np.bool_)

    for agent_index in range(num_agents):
        relative_positions = position_array - position_array[agent_index]
        distances = np.array(
            [
                math.sqrt(sum(float(component) ** 2 for component in relative_position))
                for relative_position in relative_positions
            ],
            dtype=np.float64,
        )
        candidates = [
            other_index
            for other_index in range(num_agents)
            if other_index != agent_index
            and (neighbor_radius_m is None or distances[other_index] <= neighbor_radius_m)
        ]
        candidates.sort(
            key=lambda index: (
                round(float(distances[index]), 12),
                agent_ids[index].value,
            )
        )
        for slot, neighbor_index in enumerate(candidates[:max_neighbors]):
            neighbors[agent_index, slot, :3] = relative_positions[neighbor_index]
            neighbors[agent_index, slot, 3:] = (
                velocity_array[neighbor_index] - velocity_array[agent_index]
            )
            mask[agent_index, slot] = True

    return LocalObservations(agent_ids, ego, neighbors, mask)


def build_centralized_state(
    positions: ArrayLike,
    velocities: ArrayLike,
    targets: ArrayLike,
) -> CentralizedState:
    """Build the critic state shared by kinematic and rigid-body adapters."""
    position_array, target_array = matching_point_arrays(positions, targets)
    velocity_array, _ = matching_point_arrays(velocities, positions)
    values = np.concatenate(
        (position_array.reshape(-1), velocity_array.reshape(-1), target_array.reshape(-1))
    ).astype(np.float32)
    return CentralizedState(values)
