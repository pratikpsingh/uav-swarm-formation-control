"""Tests for explicit kinematic reward components."""

import numpy as np
import pytest

from uav_swarm_control.configuration import RewardConfig
from uav_swarm_control.rewards import RewardBreakdown, compute_reward


def test_reward_components_match_hand_calculated_costs() -> None:
    config = RewardConfig(
        navigation_weight=2.0,
        formation_weight=2.0,
        collision_penalty=5.0,
        smoothness_weight=0.3,
        success_bonus=4.0,
    )
    targets = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    actions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    reward = compute_reward(
        positions,
        targets,
        actions,
        np.zeros((2, 3)),
        collision_distance_m=0.2,
        success=True,
        config=config,
    )

    np.testing.assert_allclose(reward.navigation, [-2.0, 0.0])
    np.testing.assert_allclose(reward.formation, [-0.125, -0.125])
    np.testing.assert_allclose(reward.collision, [0.0, 0.0])
    np.testing.assert_allclose(reward.smoothness, [-0.1, 0.0])
    np.testing.assert_allclose(reward.termination, [4.0, 4.0])
    np.testing.assert_allclose(reward.total, [1.775, 3.875])
    assert reward.mean_metrics()["reward/total_mean"] == pytest.approx(2.825)


def test_collision_penalty_matches_collided_agent_mask() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])
    reward = compute_reward(
        positions,
        positions,
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        collision_distance_m=0.2,
        success=False,
        config=RewardConfig(collision_penalty=3.0),
    )

    np.testing.assert_allclose(reward.collision, [-3.0, -3.0, 0.0])


def test_reward_breakdown_requires_matching_component_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        RewardBreakdown(
            navigation=np.zeros(2, dtype=np.float32),
            formation=np.zeros(1, dtype=np.float32),
            collision=np.zeros(2, dtype=np.float32),
            smoothness=np.zeros(2, dtype=np.float32),
            termination=np.zeros(2, dtype=np.float32),
        )
