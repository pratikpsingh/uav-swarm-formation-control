"""Reward-independent evaluation for the continuous reference task."""

from dataclasses import dataclass

import numpy as np
import torch

from uav_swarm_control.configuration import ContinuousBanditConfig
from uav_swarm_control.environments.continuous_bandit import ContinuousTargetBandit
from uav_swarm_control.models import ActorCritic


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Aggregate deterministic-policy metrics."""

    mean_reward: float
    action_mse: float
    success_rate: float
    episodes: int


def evaluate_continuous_bandit(
    model: ActorCritic,
    task: ContinuousBanditConfig,
    *,
    seed: int,
    device: torch.device | None = None,
) -> PolicyEvaluation:
    """Evaluate without exploration on a separate deterministic target stream."""
    evaluation_device = device or next(model.parameters()).device
    environment = ContinuousTargetBandit(task)
    observation = environment.reset(seed=seed).observation
    rewards: list[float] = []
    squared_errors: list[float] = []
    successes: list[float] = []
    was_training = model.training
    model.eval()
    for episode in range(task.evaluation_episodes):
        observation_tensor = torch.as_tensor(
            np.array(observation, copy=True),
            dtype=torch.float32,
            device=evaluation_device,
        )
        with torch.no_grad():
            action = model.deterministic_action(observation_tensor.unsqueeze(0))[0]
        transition = environment.step(action.cpu().numpy().astype(np.float32, copy=False))
        rewards.append(transition.reward)
        squared_errors.append(transition.metrics["squared_error"])
        successes.append(transition.metrics["success"])
        if episode + 1 < task.evaluation_episodes:
            observation = environment.reset().observation
    model.train(was_training)
    return PolicyEvaluation(
        mean_reward=float(np.mean(rewards)),
        action_mse=float(np.mean(squared_errors)),
        success_rate=float(np.mean(successes)),
        episodes=task.evaluation_episodes,
    )
