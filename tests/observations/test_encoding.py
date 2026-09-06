"""Tests for deterministic local actor-input encoding."""

import numpy as np
import pytest

from uav_swarm_control.agents import sequential_agent_ids
from uav_swarm_control.observations import (
    LocalObservations,
    encode_local_observations,
    encoded_local_observation_size,
)


def test_encoding_preserves_agent_rows_and_appends_neighbor_masks() -> None:
    observations = LocalObservations(
        sequential_agent_ids(2),
        ego=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        neighbors=np.array([[[5.0], [0.0]], [[6.0], [7.0]]], dtype=np.float32),
        neighbor_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
    )

    encoded = encode_local_observations(observations)

    np.testing.assert_array_equal(
        encoded,
        np.array(
            [[1.0, 2.0, 5.0, 0.0, 1.0, 0.0], [3.0, 4.0, 6.0, 7.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    assert encoded.shape == (2, encoded_local_observation_size(2, 2, 1))
    assert not encoded.flags.writeable


def test_encoded_size_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        encoded_local_observation_size(0, 2, 1)


def test_encoding_appends_obstacle_features_and_validity_mask() -> None:
    observations = LocalObservations(
        sequential_agent_ids(1),
        ego=np.array([[1.0]], dtype=np.float32),
        neighbors=np.zeros((1, 0, 6), dtype=np.float32),
        neighbor_mask=np.zeros((1, 0), dtype=np.bool_),
        obstacles=np.array([[[2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4], [0.0] * 7]], dtype=np.float32),
        obstacle_mask=np.array([[True, False]], dtype=np.bool_),
    )

    encoded = encode_local_observations(observations)

    np.testing.assert_array_equal(
        encoded,
        np.array(
            [[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4, *([0.0] * 7), 1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert encoded.shape == (1, encoded_local_observation_size(1, 0, 6, 2, 7))
