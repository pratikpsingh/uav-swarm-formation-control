"""Tests that recurrent MAPPO batching preserves contiguous temporal order."""

import torch

from uav_swarm_control.algorithms.mappo.recurrent_buffer import RecurrentMAPPORolloutBuffer
from uav_swarm_control.models.recurrent_actor_critic import RecurrentState


def test_recurrent_buffer_builds_environment_local_sequences() -> None:
    buffer = RecurrentMAPPORolloutBuffer(
        time_steps=4,
        num_environments=2,
        num_agents=2,
        local_observation_size=1,
        centralized_state_size=1,
        action_size=4,
        recurrent_layers=1,
        recurrent_hidden_size=3,
        device=torch.device("cpu"),
    )
    for time in range(4):
        local = torch.tensor([[[time * 10.0]], [[time * 10.0 + 1.0]]]).expand(2, 2, 1)
        actor = RecurrentState(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3))
        critic = RecurrentState(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
        buffer.add(
            local_observations=local,
            centralized_states=torch.zeros(2, 1),
            latent_actions=torch.zeros(2, 2, 4),
            rewards=torch.ones(2),
            log_probabilities=torch.zeros(2, 2),
            values=torch.zeros(2),
            next_values=torch.zeros(2),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            keep_masks=torch.ones(2, dtype=torch.bool),
            actor_state=actor,
            critic_state=critic,
        )

    batch = buffer.as_sequences(sequence_length=2, gamma=0.99, gae_lambda=0.95)

    assert batch.local_observations.shape == (4, 2, 2, 1)
    torch.testing.assert_close(batch.local_observations[0, :, 0, 0], torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(batch.local_observations[1, :, 0, 0], torch.tensor([20.0, 30.0]))
    selected = batch.select(torch.tensor([0, 2]))
    actor_inputs, masks = selected.actor_inputs()
    assert actor_inputs.shape == (2, 4, 1)
    assert masks.shape == (2, 4)
