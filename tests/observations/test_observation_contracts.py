"""Tests for actor observations and centralized critic state."""

import numpy as np
import pytest

from uav_swarm_control.agents import AgentId, sequential_agent_ids
from uav_swarm_control.observations import CentralizedState, LocalObservations


def _local_observations() -> LocalObservations:
    return LocalObservations(
        agent_ids=sequential_agent_ids(2),
        ego=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        neighbors=np.array(
            [[[1.0, 0.0], [0.0, 0.0]], [[-1.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        neighbor_mask=np.array([[True, False], [True, False]], dtype=np.bool_),
    )


def test_local_observations_have_canonical_shapes_and_dtypes() -> None:
    observations = _local_observations()

    assert observations.num_agents == 2
    assert observations.max_neighbors == 2
    assert observations.ego.shape == (2, 2)
    assert observations.neighbors.shape == (2, 2, 2)
    assert observations.neighbor_mask.shape == (2, 2)
    assert observations.ego.dtype == np.float32
    assert observations.neighbors.dtype == np.float32
    assert observations.neighbor_mask.dtype == np.bool_


def test_contract_arrays_are_copied_and_immutable() -> None:
    source = np.array([[1.0], [2.0]], dtype=np.float32)
    observations = LocalObservations(
        agent_ids=sequential_agent_ids(2),
        ego=source,
        neighbors=np.zeros((2, 0, 1), dtype=np.float32),
        neighbor_mask=np.zeros((2, 0), dtype=np.bool_),
    )
    source[0, 0] = 99.0

    assert observations.ego[0, 0] == 1.0
    with pytest.raises(ValueError):
        observations.ego[0, 0] = 5.0


def test_masked_neighbor_slots_must_be_zero() -> None:
    with pytest.raises(ValueError, match="zero padding"):
        LocalObservations(
            agent_ids=sequential_agent_ids(1),
            ego=np.zeros((1, 1), dtype=np.float32),
            neighbors=np.ones((1, 1, 1), dtype=np.float32),
            neighbor_mask=np.array([[False]], dtype=np.bool_),
        )


def test_neighbor_mask_requires_boolean_dtype() -> None:
    with pytest.raises(TypeError, match="bool dtype"):
        LocalObservations(
            agent_ids=sequential_agent_ids(1),
            ego=np.zeros((1, 1), dtype=np.float32),
            neighbors=np.zeros((1, 1, 1), dtype=np.float32),
            neighbor_mask=np.zeros((1, 1), dtype=np.int64),  # type: ignore[arg-type]
        )


def test_agent_rows_require_unique_matching_identifiers() -> None:
    with pytest.raises(ValueError, match="unique"):
        LocalObservations(
            agent_ids=(AgentId(0), AgentId(0)),
            ego=np.zeros((2, 1), dtype=np.float32),
            neighbors=np.zeros((2, 0, 1), dtype=np.float32),
            neighbor_mask=np.zeros((2, 0), dtype=np.bool_),
        )


def test_centralized_state_is_finite_one_dimensional_float32() -> None:
    state = CentralizedState(np.array([1.0, 2.0], dtype=np.float32))

    assert state.size == 2
    assert state.values.dtype == np.float32
    with pytest.raises(ValueError, match="finite"):
        CentralizedState(np.array([np.nan], dtype=np.float32))
