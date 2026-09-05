"""Numerically stable bounded-Gaussian operations shared by policy models."""

import math

from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional


def diagonal_normal(means: Tensor, log_standard_deviation: Tensor) -> Normal:
    """Construct a diagonal Gaussian whose scale broadcasts over leading axes."""
    return Normal(means, log_standard_deviation.exp().expand_as(means))


def squashed_log_probability(distribution: Normal, latent_actions: Tensor) -> Tensor:
    """Return the stable log density after applying tanh to a Gaussian sample."""
    log_jacobian = 2.0 * (
        math.log(2.0) - latent_actions - functional.softplus(-2.0 * latent_actions)
    )
    return (distribution.log_prob(latent_actions) - log_jacobian).sum(dim=-1)
