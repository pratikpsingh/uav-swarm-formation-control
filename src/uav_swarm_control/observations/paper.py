"""Observation encodings used by the manuscript architecture."""

from typing import Protocol, runtime_checkable

import numpy as np

from uav_swarm_control.environments.contracts import MultiAgentEnvironment
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.observations.contracts import LocalObservations


@runtime_checkable
class PositionedEnvironment(MultiAgentEnvironment, Protocol):
    """Environment exposing positions for the paper-parity actor profile."""

    @property
    def positions(self) -> FloatArray: ...


def encode_paper_flat_observations(
    observations: LocalObservations,
    environment: MultiAgentEnvironment,
) -> np.ndarray:
    """Encode ``position, velocity, target displacement, 6K neighbors``.

    Neighbor masks are intentionally omitted to reproduce the manuscript's
    ``9 + 6K`` width; invalid slots are already zero padded. Obstacle features
    and their masks are appended when an obstacle experiment is active.
    """
    if not isinstance(environment, PositionedEnvironment):
        raise TypeError("paper-flat observations require an environment.positions property.")
    if observations.ego.shape[1] != 6 or observations.neighbors.shape[2] != 6:
        raise ValueError("paper-flat encoding expects six ego and six neighbor features.")
    positions = np.asarray(environment.positions, dtype=np.float32)
    if positions.shape != (observations.num_agents, 3):
        raise ValueError("environment positions must have shape [agents, 3].")
    assert observations.obstacles is not None
    assert observations.obstacle_mask is not None
    encoded = np.concatenate(
        (
            positions,
            observations.ego[:, 3:6],
            observations.ego[:, :3],
            observations.neighbors.reshape(observations.num_agents, -1),
            observations.obstacles.reshape(observations.num_agents, -1),
            observations.obstacle_mask.astype(np.float32),
        ),
        axis=1,
        dtype=np.float32,
    )
    encoded.setflags(write=False)
    return encoded


def paper_flat_observation_size(max_neighbors: int, max_obstacles: int = 0) -> int:
    """Return the paper profile width, extended by masked obstacle records."""
    if max_neighbors < 0 or max_obstacles < 0:
        raise ValueError("observation capacities must be non-negative.")
    return 9 + 6 * max_neighbors + 8 * max_obstacles


__all__ = ["encode_paper_flat_observations", "paper_flat_observation_size"]
