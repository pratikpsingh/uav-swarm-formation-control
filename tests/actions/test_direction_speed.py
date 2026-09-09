"""Tests for the paper's four-output action adapter."""

import numpy as np
import pytest

from uav_swarm_control.actions import DirectionSpeedActions, direction_speed_to_normalized_velocity
from uav_swarm_control.agents import sequential_agent_ids


def test_direction_speed_maps_direction_and_speed_fraction() -> None:
    actions = DirectionSpeedActions(
        sequential_agent_ids(3),
        np.array(
            [[3.0 / 5.0, 4.0 / 5.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0], [0.0] * 4],
            dtype=np.float32,
        ),
    )

    converted = direction_speed_to_normalized_velocity(actions)

    np.testing.assert_allclose(converted.values[0], [0.6, 0.8, 0.0])
    np.testing.assert_array_equal(converted.values[1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(converted.values[2], [0.0, 0.0, 0.0])


def test_direction_speed_rejects_wrong_shape_and_epsilon() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        DirectionSpeedActions(sequential_agent_ids(1), np.zeros((1, 3), dtype=np.float32))
    actions = DirectionSpeedActions(sequential_agent_ids(1), np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="epsilon"):
        direction_speed_to_normalized_velocity(actions, direction_epsilon=0.0)
