"""Tests for shared decentralized actors and the centralized critic."""

import torch

from uav_swarm_control.models import SharedActorCentralCritic


def _model() -> SharedActorCentralCritic:
    torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    return SharedActorCentralCritic(5, 9, 3, 3, (16, 16), (32, 32), -0.5)


def test_shared_actor_preserves_environment_and_agent_axes() -> None:
    model = _model()
    local = torch.randn(2, 3, 5)

    output = model.sample_actions(local)

    assert output.actions.shape == (2, 3, 3)
    assert output.latent_actions.shape == (2, 3, 3)
    assert output.log_probabilities.shape == (2, 3)
    assert output.entropies.shape == (2, 3)
    assert torch.all(output.actions.abs() <= 1.0)


def test_one_shared_actor_is_equivariant_to_agent_row_permutation() -> None:
    model = _model()
    local = torch.randn(1, 3, 5)
    permutation = torch.tensor([2, 0, 1])

    original = model.deterministic_actions(local)
    permuted = model.deterministic_actions(local[:, permutation])

    torch.testing.assert_close(permuted, original[:, permutation])


def test_rollout_log_probabilities_are_exactly_recomputed() -> None:
    model = _model()
    local = torch.randn(2, 3, 5)
    sampled = model.sample_actions(local)

    recomputed = model.evaluate_latent_actions(local, sampled.latent_actions)

    torch.testing.assert_close(recomputed.log_probabilities, sampled.log_probabilities)


def test_actor_and_centralized_critic_have_separate_inputs() -> None:
    model = _model()
    local = torch.randn(3, 5)
    first_state = torch.zeros(3, 9)
    second_state = torch.ones(3, 9)

    actor_actions = model.deterministic_actions(local)
    first_values = model.values(first_state)
    second_values = model.values(second_state)

    torch.testing.assert_close(actor_actions, model.deterministic_actions(local))
    assert not torch.equal(first_values, second_values)
    assert first_values.shape == (3, 3)


def test_flattened_critic_selects_the_requested_agent_value() -> None:
    model = _model()
    states = torch.randn(4, 9)
    agent_indices = torch.tensor([0, 2, 1, 0], dtype=torch.int64)

    all_values = model.values(states)
    selected = model.values_for_agents(states, agent_indices)

    expected = all_values[torch.arange(4), agent_indices]
    torch.testing.assert_close(selected, expected)
