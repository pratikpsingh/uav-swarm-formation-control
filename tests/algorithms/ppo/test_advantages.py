"""Hand-calculated tests for discounted returns and GAE."""

import pytest
import torch

from uav_swarm_control.algorithms.ppo import (
    discounted_returns,
    generalized_advantage_estimate,
)


def test_discounted_returns_stop_at_true_terminal_states() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    next_values = torch.tensor([0.0, 20.0, 10.0])
    terminated = torch.tensor([False, True, False])
    truncated = torch.tensor([False, False, True])

    actual = discounted_returns(
        rewards,
        next_values,
        terminated,
        truncated,
        gamma=0.5,
    )

    torch.testing.assert_close(actual, torch.tensor([2.0, 2.0, 8.0]))


def test_gae_matches_two_step_hand_calculation() -> None:
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.5])
    next_values = torch.tensor([0.5, 10.0])
    terminated = torch.tensor([False, True])
    truncated = torch.tensor([False, False])

    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([1.5, 0.5]))
    torch.testing.assert_close(returns, torch.tensor([2.0, 1.0]))


def test_gae_bootstraps_at_truncation_but_does_not_cross_it() -> None:
    advantages, returns = generalized_advantage_estimate(
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([2.0]),
        torch.tensor([False]),
        torch.tensor([True]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantages.item() == pytest.approx(2.3)
    assert returns.item() == pytest.approx(2.8)


def test_return_estimators_reject_mismatched_vectors() -> None:
    with pytest.raises(ValueError, match="equal length"):
        generalized_advantage_estimate(
            torch.tensor([1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            gamma=0.9,
            gae_lambda=0.95,
        )
