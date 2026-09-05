"""Return and advantage estimators with explicit episode-boundary semantics."""

import torch
from torch import Tensor


def _validate_trajectories(*trajectories: Tensor) -> None:
    if not trajectories or trajectories[0].ndim < 1:
        raise ValueError("return-estimator inputs must include a time dimension.")
    shape = trajectories[0].shape
    if shape[0] == 0 or any(trajectory.shape != shape for trajectory in trajectories):
        raise ValueError("return-estimator inputs must be non-empty tensors of equal shape.")


def discounted_returns(
    rewards: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Compute returns, bootstrapping without crossing an episode boundary."""
    _validate_trajectories(rewards, next_values, terminated, truncated)
    returns = torch.empty_like(rewards)
    running = next_values[-1]
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrapped = rewards[index] + gamma * next_values[index]
        continued = rewards[index] + gamma * running
        boundary_value = (
            bootstrapped
            if index == rewards.shape[0] - 1
            else torch.where(
                truncated[index],
                bootstrapped,
                continued,
            )
        )
        running = torch.where(terminated[index], rewards[index], boundary_value)
        returns[index] = running
    return returns


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE and value targets while distinguishing terminal and time-limit steps."""
    _validate_trajectories(rewards, values, next_values, terminated, truncated)
    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = (~terminated[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        continue_episode = (~terminated[index] & ~truncated[index]).to(rewards.dtype)
        running = delta + gamma * gae_lambda * continue_episode * running
        advantages[index] = running
    return advantages, advantages + values
