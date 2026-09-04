"""Tests for the deterministic proportional controller."""

import numpy as np

from uav_swarm_control.agents import sequential_agent_ids
from uav_swarm_control.controllers import ProportionalPositionController
from uav_swarm_control.observations import LocalObservations


def test_proportional_controller_scales_and_clips_each_velocity_component() -> None:
    observations = LocalObservations(
        sequential_agent_ids(2),
        np.array([[4.0, -1.0, 0.5, 9.0], [0.0, 0.0, 0.0, 9.0]], dtype=np.float32),
        np.zeros((2, 0, 1), dtype=np.float32),
        np.zeros((2, 0), dtype=np.bool_),
    )
    controller = ProportionalPositionController(
        gain_per_second=2.0,
        max_velocity_component_mps=4.0,
    )

    actions = controller.act(observations)

    np.testing.assert_allclose(
        actions.values,
        [[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]],
    )
