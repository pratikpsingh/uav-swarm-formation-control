"""Pure PPO loss calculation, separated for review and hand testing."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PPOLoss:
    """Total differentiable objective and diagnostic components."""

    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def ppo_loss(
    *,
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    new_values: Tensor,
    returns: Tensor,
    entropies: Tensor,
    clip_coefficient: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> PPOLoss:
    """Calculate clipped policy, value, and exploration objectives."""
    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_coefficient, 1.0 + clip_coefficient) * advantages
    policy = -torch.minimum(unclipped, clipped).mean()
    value = 0.5 * torch.mean((new_values - returns).square())
    entropy = entropies.mean()
    total = policy + value_coefficient * value - entropy_coefficient * entropy
    with torch.no_grad():
        approximate_kl = (old_log_probabilities - new_log_probabilities).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coefficient).float().mean()
    return PPOLoss(total, policy, value, entropy, approximate_kl, clip_fraction)
