"""Hand-checkable dynamics and behavior tests for the DMPC adaptation."""

from pathlib import Path

import numpy as np
import pytest

from uav_swarm_control.agents import sequential_agent_ids
from uav_swarm_control.configuration import load_dmpc_config
from uav_swarm_control.controllers.dmpc import (
    DistributedMPCController,
    prediction_matrices,
    triple_integrator_matrices,
)
from uav_swarm_control.observations import CentralizedState, LocalObservations

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/algorithm/dmpc.yaml"


def test_triple_integrator_matches_constant_jerk_equations() -> None:
    state_matrix, input_matrix = triple_integrator_matrices(0.2)
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    jerk = np.array([1.0, 0.0, 0.0])
    next_state = state_matrix @ state + input_matrix @ jerk
    np.testing.assert_allclose(
        next_state,
        [0.2 + 0.2**3 / 6, 0, 0, 1 + 0.2**2 / 2, 0, 0, 0.2, 0, 0],
    )


def test_stacked_prediction_matches_repeated_state_updates() -> None:
    free, control = prediction_matrices(2, 0.2)
    state_matrix, input_matrix = triple_integrator_matrices(0.2)
    initial = np.array([0.0, 0, 0, 1.0, 0, 0, 0.0, 0, 0])
    jerks = np.array([[1.0, 0, 0], [0.0, 0, 0]])
    first = state_matrix @ initial + input_matrix @ jerks[0]
    second = state_matrix @ first + input_matrix @ jerks[1]
    predicted = (free @ initial + control @ jerks.reshape(-1)).reshape(2, 9)
    np.testing.assert_allclose(predicted, [first, second])


def test_controller_moves_translated_formation_toward_targets_and_resets_diagnostics() -> None:
    identifiers = sequential_agent_ids(2)
    positions = np.array([[-0.5, 0, 1], [0.5, 0, 1]], dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    targets = positions + np.array([1.0, 0, 0], dtype=np.float32)
    observations = LocalObservations(
        identifiers,
        np.concatenate((targets - positions, velocities), axis=1),
        np.zeros((2, 0, 6), dtype=np.float32),
        np.zeros((2, 0), dtype=np.bool_),
    )
    state = CentralizedState(
        np.concatenate((positions.reshape(-1), velocities.reshape(-1), targets.reshape(-1)))
    )
    controller = DistributedMPCController(
        load_dmpc_config(CONFIG),
        control_time_step_seconds=1 / 30,
        max_velocity_component_mps=0.5,
        max_neighbors=0,
        neighbor_radius_m=None,
    )

    actions = controller.act_with_state(observations, state)

    assert np.all(actions.values[:, 0] > 0)
    np.testing.assert_allclose(actions.values[:, 1:], 0, atol=1e-8)
    assert controller.diagnostics()["optimization_runs"] == 2
    assert controller.diagnostics()["solver_success_rate"] == pytest.approx(1)
    controller.reset()
    assert controller.diagnostics()["optimization_runs"] == 0


@pytest.mark.parametrize(
    ("control_step", "planning_step"),
    [(0.03, 0.2), (1 / 30, 0.15)],
)
def test_controller_rejects_incommensurate_control_rates(
    control_step: float, planning_step: float
) -> None:
    config = load_dmpc_config(CONFIG)
    config = type(config)(
        config.schema_version,
        config.name,
        config.prediction_horizon_steps,
        planning_step,
        config.weights,
        config.limits,
        config.solver,
    )
    with pytest.raises(ValueError, match="integer multiple"):
        DistributedMPCController(
            config,
            control_time_step_seconds=control_step,
            max_velocity_component_mps=0.5,
            max_neighbors=2,
            neighbor_radius_m=None,
        )
