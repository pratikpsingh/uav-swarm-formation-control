"""Reviewable single-agent Proximal Policy Optimization building blocks."""

from uav_swarm_control.algorithms.ppo.advantages import (
    discounted_returns,
    generalized_advantage_estimate,
)
from uav_swarm_control.algorithms.ppo.buffer import RolloutBatch, RolloutBuffer
from uav_swarm_control.algorithms.ppo.checkpoint import (
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from uav_swarm_control.algorithms.ppo.evaluation import (
    PolicyEvaluation,
    evaluate_continuous_bandit,
)
from uav_swarm_control.algorithms.ppo.objectives import PPOLoss, ppo_loss
from uav_swarm_control.algorithms.ppo.trainer import (
    PPOTrainingResult,
    PPOUpdateMetrics,
    resolve_device,
    seed_torch,
    train_ppo,
)

__all__ = [
    "PPOLoss",
    "PPOTrainingResult",
    "PPOUpdateMetrics",
    "PolicyEvaluation",
    "RolloutBatch",
    "RolloutBuffer",
    "discounted_returns",
    "evaluate_continuous_bandit",
    "generalized_advantage_estimate",
    "load_policy_checkpoint",
    "ppo_loss",
    "resolve_device",
    "save_policy_checkpoint",
    "seed_torch",
    "train_ppo",
]
