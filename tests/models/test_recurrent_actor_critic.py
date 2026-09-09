"""Shape, memory, reset, and information-boundary tests for the recurrent model."""

import torch

from uav_swarm_control.models.recurrent_actor_critic import PaperRecurrentActorCritic


def test_recurrent_actor_and_scalar_critic_shapes() -> None:
    model = PaperRecurrentActorCritic(9, 18, feature_size=12, recurrent_hidden_size=10)
    actor_state = model.initial_actor_state(6, device=torch.device("cpu"))
    critic_state = model.initial_critic_state(2, device=torch.device("cpu"))
    local = torch.randn(4, 6, 9)
    states = torch.randn(4, 2, 18)
    actor_masks = torch.ones(4, 6, dtype=torch.bool)
    critic_masks = torch.ones(4, 2, dtype=torch.bool)

    policy = model.sample_actions(local, actor_state, actor_masks)
    values, final_critic = model.values(states, critic_state, critic_masks)

    assert policy.actions.shape == (4, 6, 4)
    assert policy.log_probabilities.shape == (4, 6)
    assert policy.state.hidden.shape == (1, 6, 10)
    assert values.shape == (4, 2)
    assert final_critic.hidden.shape == (1, 2, 10)


def test_sequence_history_changes_output_and_reset_removes_history() -> None:
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    model = PaperRecurrentActorCritic(3, 6, feature_size=8, recurrent_hidden_size=8)
    initial = model.initial_actor_state(1, device=torch.device("cpu"))
    history = torch.tensor([[[1.0, -1.0, 0.5]]])
    current = torch.tensor([[[0.2, 0.1, -0.3]]])
    keep = torch.ones((1, 1), dtype=torch.bool)
    _, remembered = model.deterministic_actions(history, initial, keep)

    with_history, _ = model.deterministic_actions(current, remembered, keep)
    after_reset, _ = model.deterministic_actions(current, remembered, torch.zeros_like(keep))
    direct, _ = model.deterministic_actions(current, initial, keep)

    assert not torch.allclose(with_history, direct)
    torch.testing.assert_close(after_reset, direct)
