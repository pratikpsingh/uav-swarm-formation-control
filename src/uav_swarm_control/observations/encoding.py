"""Deterministic tensor-ready encodings of structured local observations."""

import numpy as np

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.observations.contracts import LocalObservations


def encoded_local_observation_size(
    ego_features: int,
    max_neighbors: int,
    neighbor_features: int,
) -> int:
    """Return the flat actor-input width, including explicit validity masks."""
    if ego_features < 1 or max_neighbors < 0 or neighbor_features < 1:
        raise ValueError("observation dimensions must be non-negative and feature sizes positive.")
    return ego_features + max_neighbors * neighbor_features + max_neighbors


def encode_local_observations(observations: LocalObservations) -> Float32Array:
    """Flatten neighbor slots and append their mask without mixing agent rows."""
    num_agents = observations.num_agents
    encoded = np.concatenate(
        (
            observations.ego,
            observations.neighbors.reshape(num_agents, -1),
            observations.neighbor_mask.astype(np.float32),
        ),
        axis=1,
        dtype=np.float32,
    )
    encoded.setflags(write=False)
    return encoded
