"""Tests for fixed-size on-policy storage."""

import pytest
import torch

from uav_swarm_control.algorithms.ppo import RolloutBuffer


def _add_terminal_transition(buffer: RolloutBuffer, value: float) -> None:
    buffer.add(
        observation=torch.tensor([value]),
        latent_action=torch.tensor([0.0]),
        reward=value,
        log_probability=torch.tensor(-0.5),
        value=torch.tensor(0.0),
        next_value=torch.tensor(100.0),
        terminated=True,
        truncated=False,
    )


def test_buffer_computes_returns_and_normalizes_advantages() -> None:
    buffer = RolloutBuffer(2, 1, 1, torch.device("cpu"))
    _add_terminal_transition(buffer, 1.0)
    _add_terminal_transition(buffer, 3.0)

    batch = buffer.as_batch(gamma=0.99, gae_lambda=0.95)

    torch.testing.assert_close(batch.returns, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([-1.0, 1.0]))


def test_buffer_rejects_incomplete_and_overfull_rollouts() -> None:
    buffer = RolloutBuffer(1, 1, 1, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="must be full"):
        buffer.as_batch(gamma=0.99, gae_lambda=0.95)

    _add_terminal_transition(buffer, 1.0)
    with pytest.raises(RuntimeError, match="is full"):
        _add_terminal_transition(buffer, 2.0)
