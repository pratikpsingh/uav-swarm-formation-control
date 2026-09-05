"""A tiny continuous-control task for testing an RL algorithm in isolation."""

import numpy as np

from uav_swarm_control._arrays import Float32Array, immutable_float32_array
from uav_swarm_control.configuration import ContinuousBanditConfig
from uav_swarm_control.environments.single_agent import SingleAgentReset, SingleAgentStep
from uav_swarm_control.seeding import RandomStream, make_rng


class ContinuousTargetBandit:
    """Choose an action matching the observed target in one environment step."""

    def __init__(self, config: ContinuousBanditConfig) -> None:
        self._config = config
        self._rng: np.random.Generator | None = None
        self._target: float | None = None
        self._done = True

    @property
    def observation_size(self) -> int:
        return 1

    @property
    def action_size(self) -> int:
        return 1

    def reset(self, *, seed: int | None = None) -> SingleAgentReset:
        """Sample a target; a seed replaces the task's random stream."""
        if seed is not None:
            self._rng = make_rng(seed, RandomStream.ENVIRONMENT)
        if self._rng is None:
            raise RuntimeError("the first reset must provide a seed.")
        self._target = float(self._rng.uniform(self._config.target_low, self._config.target_high))
        self._done = False
        return SingleAgentReset(np.array([self._target], dtype=np.float32))

    def step(self, action: Float32Array) -> SingleAgentStep:
        """Score one bounded action and terminate the one-step episode."""
        if self._target is None or self._done:
            raise RuntimeError("reset must be called before step.")
        checked = immutable_float32_array(action, name="action", dimensions=1)
        if checked.shape != (1,):
            raise ValueError(f"action must have shape (1,); received {checked.shape}.")
        selected = float(checked[0])
        if not -1.0 <= selected <= 1.0:
            raise ValueError("action values must lie in [-1, 1].")
        error = selected - self._target
        squared_error = error * error
        self._done = True
        return SingleAgentStep(
            observation=np.array([self._target], dtype=np.float32),
            reward=float(1.0 - squared_error),
            terminated=True,
            truncated=False,
            metrics={
                "target": self._target,
                "action": selected,
                "absolute_error": abs(error),
                "squared_error": squared_error,
                "success": float(abs(error) <= self._config.success_tolerance),
            },
        )
