"""Hand-calculated tests for the clipped PPO objective."""

import math

import pytest
import torch

from uav_swarm_control.algorithms.ppo import ppo_loss


def test_ppo_loss_matches_hand_calculation() -> None:
    old = torch.zeros(2)
    new = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)

    loss = ppo_loss(
        new_log_probabilities=new,
        old_log_probabilities=old,
        advantages=torch.tensor([1.0, -1.0]),
        new_values=torch.tensor([0.0, 1.0]),
        returns=torch.tensor([1.0, 1.0]),
        entropies=torch.ones(2),
        clip_coefficient=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    assert loss.policy.item() == pytest.approx(-0.2)
    assert loss.value.item() == pytest.approx(0.25)
    assert loss.entropy.item() == pytest.approx(1.0)
    assert loss.total.item() == pytest.approx(-0.085)
    assert loss.clip_fraction.item() == pytest.approx(1.0)
    loss.total.backward()  # pyright: ignore[reportUnknownMemberType]
    assert new.grad is not None
