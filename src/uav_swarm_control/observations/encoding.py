"""Deterministic tensor-ready encodings of structured local observations."""

import numpy as np

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.observations.contracts import LocalObservations


def encoded_local_observation_size(
    ego_features: int,
    max_neighbors: int,
    neighbor_features: int,
    max_obstacles: int = 0,
    obstacle_features: int = 7,
) -> int:
    """Return the flat actor-input width, including explicit validity masks."""
    if (
        ego_features < 1
        or max_neighbors < 0
        or neighbor_features < 1
        or max_obstacles < 0
        or obstacle_features < 1
    ):
        raise ValueError("observation dimensions must be non-negative and feature sizes positive.")
    return (
        ego_features
        + max_neighbors * neighbor_features
        + max_neighbors
        + max_obstacles * obstacle_features
        + max_obstacles
    )


def encode_local_observations(observations: LocalObservations) -> Float32Array:
    """Flatten neighbor/obstacle slots and masks without mixing agent rows."""
    num_agents = observations.num_agents
    assert observations.obstacles is not None
    assert observations.obstacle_mask is not None
    encoded = np.concatenate(
        (
            observations.ego,
            observations.neighbors.reshape(num_agents, -1),
            observations.neighbor_mask.astype(np.float32),
            observations.obstacles.reshape(num_agents, -1),
            observations.obstacle_mask.astype(np.float32),
        ),
        axis=1,
        dtype=np.float32,
    )
    encoded.setflags(write=False)
    return encoded
