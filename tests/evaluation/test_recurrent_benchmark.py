"""Regression tests for recurrent episode evaluation metadata."""

import numpy as np

from uav_swarm_control.agents import AgentId, sequential_agent_ids
from uav_swarm_control.environments import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.evaluation.recurrent_benchmark import (
    evaluate_recurrent_environment,
)
from uav_swarm_control.observations import CentralizedState, LocalObservations


class _Controller:
    def reset(self) -> None:
        pass

    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        return NormalizedVelocityActions(
            observations.agent_ids,
            np.zeros((observations.num_agents, 3), dtype=np.float32),
        )


class _Environment:
    def __init__(self) -> None:
        self._agent_ids = sequential_agent_ids(2)
        self._positions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        return self._agent_ids

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    def _observations(self) -> LocalObservations:
        return LocalObservations(
            self.agent_ids,
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 1, 3), dtype=np.float32),
            np.zeros((2, 1), dtype=np.bool_),
        )

    def reset(self, *, seed: int) -> ResetResult:
        del seed
        return ResetResult(self._observations(), CentralizedState(np.zeros(6, dtype=np.float32)))

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        assert actions.agent_ids == self.agent_ids
        return StepResult(
            observations=self._observations(),
            centralized_state=CentralizedState(np.zeros(6, dtype=np.float32)),
            rewards=np.zeros(2, dtype=np.float32),
            terminated=True,
            truncated=False,
            metrics={
                "success": 1.0,
                "collision_pairs": 0.0,
                "minimum_separation_m": 1.0,
                "normalized_shape_rmse": 0.0,
                "position_rmse_m": 0.0,
            },
        )

    def close(self) -> None:
        pass


def test_evaluation_records_episode_context() -> None:
    environment = _Environment()

    records = evaluate_recurrent_environment(
        lambda: environment,
        _Controller(),
        episodes=1,
        seed=7,
        horizon=2,
        time_step_seconds=0.1,
        collision_distance_m=0.2,
        episode_metadata=lambda _: {"formation_index": 3.0},
    )

    assert records[0]["context/formation_index"] == 3.0
    assert records[0]["collision_free_success"] == 1.0
