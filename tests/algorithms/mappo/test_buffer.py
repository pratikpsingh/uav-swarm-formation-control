"""Tests for explicit MAPPO rollout axes and flattening alignment."""

import torch

from uav_swarm_control.algorithms.mappo import MAPPORolloutBuffer


def test_time_environment_agent_axes_flatten_without_misalignment() -> None:
    buffer = MAPPORolloutBuffer(
        time_steps=2,
        num_environments=2,
        num_agents=3,
        local_observation_size=1,
        centralized_state_size=1,
        action_size=1,
        device=torch.device("cpu"),
    )
    for time in range(2):
        local = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1) + time * 100
        states = torch.tensor([[time * 10.0], [time * 10.0 + 1.0]])
        agent_values = local.squeeze(-1)
        buffer.add(
            local_observations=local,
            centralized_states=states,
            latent_actions=torch.zeros(2, 3, 1),
            rewards=agent_values,
            log_probabilities=torch.zeros(2, 3),
            values=torch.zeros(2, 3),
            next_values=torch.zeros(2, 3),
            terminated=torch.ones(2, 3, dtype=torch.bool),
            truncated=torch.zeros(2, 3, dtype=torch.bool),
        )

    batch = buffer.as_batch(gamma=0.99, gae_lambda=0.95)
    flattened = batch.flatten()

    assert batch.local_observations.shape == (2, 2, 3, 1)
    assert flattened.size == 12
    torch.testing.assert_close(
        flattened.agent_indices,
        torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
    )
    torch.testing.assert_close(
        flattened.centralized_states.squeeze(-1),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0]),
    )
    torch.testing.assert_close(flattened.returns, flattened.local_observations.squeeze(-1))
    assert torch.isclose(flattened.advantages.mean(), torch.tensor(0.0), atol=1e-6)


def test_multidimensional_gae_keeps_parallel_trajectories_independent() -> None:
    from uav_swarm_control.algorithms.ppo import generalized_advantage_estimate

    rewards = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    zeros = torch.zeros_like(rewards)
    terminated = torch.ones_like(rewards, dtype=torch.bool)

    advantages, returns = generalized_advantage_estimate(
        rewards,
        zeros,
        zeros,
        terminated,
        torch.zeros_like(terminated),
        gamma=0.99,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(advantages, rewards)
    torch.testing.assert_close(returns, rewards)
