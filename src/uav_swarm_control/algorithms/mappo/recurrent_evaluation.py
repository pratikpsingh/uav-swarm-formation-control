"""Deterministic actor-only evaluation of recurrent swarm policies."""

from dataclasses import dataclass

import numpy as np
import torch

from uav_swarm_control.actions import DirectionSpeedActions, direction_speed_to_normalized_velocity
from uav_swarm_control.algorithms.mappo.recurrent_trainer import (
    EnvironmentFactory,
    ObservationEncoder,
    observation_encoder_for_profile,
)
from uav_swarm_control.configuration.recurrent_mappo import RecurrentObservationProfile
from uav_swarm_control.models.recurrent_actor_critic import PaperRecurrentActorCritic
from uav_swarm_control.seeding import RandomStream, derive_indexed_seed


@dataclass(frozen=True, slots=True)
class RecurrentMAPPOEvaluation:
    """Held-out task metrics aggregated over deterministic episodes."""

    episodes: int
    success_rate: float
    collision_episode_rate: float
    mean_episode_return: float
    mean_episode_steps: float
    mean_final_position_rmse_m: float
    mean_final_shape_rmse_m: float


def evaluate_recurrent_mappo(
    model: PaperRecurrentActorCritic,
    environment_factory: EnvironmentFactory,
    *,
    episodes: int,
    seed: int,
    observation_profile: RecurrentObservationProfile,
    direction_epsilon: float = 1e-6,
    observation_encoder: ObservationEncoder | None = None,
    device: torch.device | None = None,
) -> RecurrentMAPPOEvaluation:
    """Evaluate with local observations and per-UAV state; the critic is never called."""
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive.")
    encoder = observation_encoder or observation_encoder_for_profile(observation_profile)
    evaluation_device = device or next(model.parameters()).device
    environment = environment_factory()
    was_training = model.training
    model.eval()
    successes: list[float] = []
    collisions: list[float] = []
    returns: list[float] = []
    steps: list[float] = []
    position_errors: list[float] = []
    shape_errors: list[float] = []
    try:
        for episode in range(episodes):
            reset = environment.reset(
                seed=derive_indexed_seed(seed, RandomStream.EVALUATION, episode)
            )
            observations = reset.observations
            state = model.initial_actor_state(len(environment.agent_ids), device=evaluation_device)
            keep = torch.zeros(
                len(environment.agent_ids), dtype=torch.bool, device=evaluation_device
            )
            episode_return = 0.0
            collision = False
            while True:
                local = torch.as_tensor(
                    np.array(encoder(observations, environment), copy=True),
                    dtype=torch.float32,
                    device=evaluation_device,
                )
                with torch.no_grad():
                    bounded, state = model.deterministic_actions(
                        local.unsqueeze(0), state, keep.unsqueeze(0)
                    )
                actions = direction_speed_to_normalized_velocity(
                    DirectionSpeedActions(
                        environment.agent_ids,
                        bounded.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
                    ),
                    direction_epsilon=direction_epsilon,
                )
                transition = environment.step(actions)
                episode_return += float(transition.rewards.mean())
                collision |= transition.metrics["collision_pairs"] > 0.0
                observations = transition.observations
                keep = torch.ones_like(keep)
                if transition.episode_done:
                    break
            successes.append(transition.metrics["success"])
            collisions.append(float(collision))
            returns.append(episode_return)
            steps.append(transition.metrics["step_count"])
            position_errors.append(transition.metrics["position_rmse_m"])
            shape_errors.append(transition.metrics["shape_rmse_m"])
    finally:
        environment.close()
        model.train(was_training)
    return RecurrentMAPPOEvaluation(
        episodes,
        float(np.mean(successes)),
        float(np.mean(collisions)),
        float(np.mean(returns)),
        float(np.mean(steps)),
        float(np.mean(position_errors)),
        float(np.mean(shape_errors)),
    )


__all__ = ["RecurrentMAPPOEvaluation", "evaluate_recurrent_mappo"]
