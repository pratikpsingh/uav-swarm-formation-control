"""Deterministic reward-independent evaluation for kinematic MAPPO."""

from dataclasses import dataclass

import numpy as np
import torch

from uav_swarm_control.algorithms.mappo.trainer import EnvironmentFactory
from uav_swarm_control.environments import NormalizedVelocityActions
from uav_swarm_control.models import SharedActorCentralCritic
from uav_swarm_control.observations import encode_local_observations
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed


@dataclass(frozen=True, slots=True)
class MAPPOEvaluation:
    """Common kinematic metrics aggregated over deterministic policy episodes."""

    episodes: int
    success_rate: float
    collision_episode_rate: float
    mean_episode_return: float
    mean_episode_steps: float
    mean_final_position_rmse_m: float
    mean_final_shape_rmse_m: float


def evaluate_mappo(
    model: SharedActorCentralCritic,
    environment_factory: EnvironmentFactory,
    *,
    episodes: int,
    seed: int,
    device: torch.device | None = None,
) -> MAPPOEvaluation:
    """Run the shared actor without exploration or centralized critic access."""
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive.")
    evaluation_device = device or next(model.parameters()).device
    successes: list[float] = []
    collisions: list[float] = []
    returns: list[float] = []
    steps: list[float] = []
    position_errors: list[float] = []
    shape_errors: list[float] = []
    environment = environment_factory()
    was_training = model.training
    model.eval()
    try:
        for episode in range(episodes):
            reset = environment.reset(
                seed=derive_indexed_seed(seed, RandomStream.EVALUATION, episode)
            )
            observations = reset.observations
            episode_return = 0.0
            episode_had_collision = False
            while True:
                local = torch.as_tensor(
                    np.array(encode_local_observations(observations), copy=True),
                    dtype=torch.float32,
                    device=evaluation_device,
                )
                with torch.no_grad():
                    actions = model.deterministic_actions(local)
                transition = environment.step(
                    NormalizedVelocityActions(
                        environment.agent_ids,
                        actions.cpu().numpy().astype(np.float32, copy=False),
                    )
                )
                episode_return += float(np.mean(transition.rewards))
                episode_had_collision |= transition.metrics["collision_pairs"] > 0.0
                observations = transition.observations
                if transition.episode_done:
                    break
            successes.append(transition.metrics["success"])
            collisions.append(float(episode_had_collision))
            returns.append(episode_return)
            steps.append(transition.metrics["step_count"])
            position_errors.append(transition.metrics["position_rmse_m"])
            shape_errors.append(transition.metrics["shape_rmse_m"])
    finally:
        environment.close()
        model.train(was_training)
    return MAPPOEvaluation(
        episodes=episodes,
        success_rate=float(np.mean(successes)),
        collision_episode_rate=float(np.mean(collisions)),
        mean_episode_return=float(np.mean(returns)),
        mean_episode_steps=float(np.mean(steps)),
        mean_final_position_rmse_m=float(np.mean(position_errors)),
        mean_final_shape_rmse_m=float(np.mean(shape_errors)),
    )
