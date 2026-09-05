"""Return and advantage estimators with explicit episode-boundary semantics."""

import torch
from torch import Tensor


def _validate_vectors(*vectors: Tensor) -> None:
    if not vectors or vectors[0].ndim != 1:
        raise ValueError("return-estimator inputs must be one-dimensional.")
    length = vectors[0].shape[0]
    if length == 0 or any(vector.ndim != 1 or vector.shape[0] != length for vector in vectors):
        raise ValueError("return-estimator inputs must be non-empty vectors of equal length.")


def discounted_returns(
    rewards: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Compute returns, bootstrapping without crossing an episode boundary."""
    _validate_vectors(rewards, next_values, terminated, truncated)
    returns = torch.empty_like(rewards)
    running = next_values[-1]
    for index in range(rewards.shape[0] - 1, -1, -1):
        if bool(terminated[index]):
            running = rewards[index]
        elif bool(truncated[index]) or index == rewards.shape[0] - 1:
            running = rewards[index] + gamma * next_values[index]
        else:
            running = rewards[index] + gamma * running
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
    _validate_vectors(rewards, values, next_values, terminated, truncated)
    advantages = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = (~terminated[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        continue_episode = (~terminated[index] & ~truncated[index]).to(rewards.dtype)
        running = delta + gamma * gae_lambda * continue_episode * running
        advantages[index] = running
    return advantages, advantages + values
