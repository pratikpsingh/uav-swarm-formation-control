"""Tests for the minimal continuous-control reference task."""

import numpy as np
import pytest

from uav_swarm_control.configuration import ContinuousBanditConfig
from uav_swarm_control.environments import ContinuousTargetBandit, SingleAgentEnvironment


def _task() -> ContinuousBanditConfig:
    return ContinuousBanditConfig(-0.8, 0.8, 0.1, 32)


def test_bandit_is_a_deterministic_single_agent_environment() -> None:
    first = ContinuousTargetBandit(_task())
    second = ContinuousTargetBandit(_task())

    first_observation = first.reset(seed=123).observation
    second_observation = second.reset(seed=123).observation

    assert isinstance(first, SingleAgentEnvironment)
    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_observation.dtype == np.float32
    assert not first_observation.flags.writeable


def test_matching_target_is_optimal_and_terminates_episode() -> None:
    environment = ContinuousTargetBandit(_task())
    target = environment.reset(seed=7).observation

    transition = environment.step(target)

    assert transition.reward == pytest.approx(1.0)
    assert transition.metrics["squared_error"] == pytest.approx(0.0)
    assert transition.metrics["success"] == 1.0
    assert transition.terminated
    assert transition.episode_done
    with pytest.raises(RuntimeError, match="reset"):
        environment.step(target)


def test_bandit_requires_initial_seed_and_bounded_action() -> None:
    environment = ContinuousTargetBandit(_task())
    with pytest.raises(RuntimeError, match="first reset"):
        environment.reset()
    environment.reset(seed=5)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        environment.step(np.array([1.1], dtype=np.float32))
