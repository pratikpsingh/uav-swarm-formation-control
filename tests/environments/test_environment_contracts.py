"""Tests for the simulator-independent environment boundary."""

from collections.abc import Mapping

import numpy as np
import pytest

from uav_swarm_control.agents import sequential_agent_ids
from uav_swarm_control.environments import NormalizedVelocityActions, StepResult
from uav_swarm_control.observations import CentralizedState, LocalObservations


def _observations() -> LocalObservations:
    return LocalObservations(
        agent_ids=sequential_agent_ids(2),
        ego=np.zeros((2, 3), dtype=np.float32),
        neighbors=np.zeros((2, 1, 3), dtype=np.float32),
        neighbor_mask=np.zeros((2, 1), dtype=np.bool_),
    )


def _step_result(
    *,
    terminated: bool = False,
    truncated: bool = False,
    metrics: Mapping[str, float] | None = None,
) -> StepResult:
    return StepResult(
        observations=_observations(),
        centralized_state=CentralizedState(np.zeros(6, dtype=np.float32)),
        rewards=np.array([1.0, 1.0], dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
        metrics=metrics or {"formation_rmse_m": 0.25},
    )


def test_normalized_velocity_actions_have_one_3d_row_per_agent() -> None:
    actions = NormalizedVelocityActions(
        sequential_agent_ids(2),
        np.array([[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]], dtype=np.float32),
    )

    assert actions.num_agents == 2
    assert actions.values.shape == (2, 3)
    assert not actions.values.flags.writeable


def test_actions_reject_values_outside_normalized_range() -> None:
    with pytest.raises(ValueError, match="closed interval"):
        NormalizedVelocityActions(
            sequential_agent_ids(1),
            np.array([[1.01, 0.0, 0.0]], dtype=np.float32),
        )


def test_step_result_distinguishes_termination_from_truncation() -> None:
    terminal = _step_result(terminated=True)
    time_limit = _step_result(truncated=True)

    assert terminal.episode_done
    assert terminal.terminated and not terminal.truncated
    assert time_limit.episode_done
    assert time_limit.truncated and not time_limit.terminated


def test_step_result_rejects_ambiguous_episode_end() -> None:
    with pytest.raises(ValueError, match="both"):
        _step_result(terminated=True, truncated=True)


def test_step_result_requires_one_reward_per_agent() -> None:
    with pytest.raises(ValueError, match="one value per agent"):
        StepResult(
            observations=_observations(),
            centralized_state=CentralizedState(np.zeros(6, dtype=np.float32)),
            rewards=np.array([1.0], dtype=np.float32),
            terminated=False,
            truncated=False,
            metrics={},
        )


def test_metrics_are_finite_and_immutable() -> None:
    source = {"formation_rmse_m": 0.25}
    result = _step_result(metrics=source)
    source["formation_rmse_m"] = 10.0

    assert result.metrics["formation_rmse_m"] == 0.25
    with pytest.raises(TypeError):
        result.metrics["collision_count"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        _step_result(metrics={"bad": float("inf")})
