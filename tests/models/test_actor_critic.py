"""Tests for the bounded continuous actor-critic."""

import torch

from uav_swarm_control.models import ActorCritic


def _model() -> ActorCritic:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return ActorCritic(2, 1, (8, 8), -0.5)


def test_actor_critic_shapes_and_action_bounds() -> None:
    model = _model()
    observations = torch.tensor([[0.2, -0.1], [0.5, 0.4]])

    output = model.sample(observations)

    assert output.action.shape == (2, 1)
    assert output.latent_action.shape == (2, 1)
    assert output.log_probability.shape == (2,)
    assert output.entropy.shape == (2,)
    assert output.value.shape == (2,)
    assert torch.all(output.action.abs() <= 1.0)


def test_rollout_log_probabilities_can_be_recomputed() -> None:
    model = _model()
    observations = torch.tensor([[0.2, -0.1], [0.5, 0.4]])
    sampled = model.sample(observations)

    recomputed = model.evaluate_latent_actions(observations, sampled.latent_action)

    torch.testing.assert_close(recomputed.log_probability, sampled.log_probability)


def test_saturated_action_log_probability_is_still_recomputable() -> None:
    model = _model()
    observations = torch.tensor([[0.2, -0.1]])
    latent_action = torch.tensor([[20.0]])

    evaluated = model.evaluate_latent_actions(observations, latent_action)
    recomputed = model.evaluate_latent_actions(observations, evaluated.latent_action)

    assert evaluated.action.item() == 1.0
    assert torch.isfinite(evaluated.log_probability).all()
    torch.testing.assert_close(recomputed.log_probability, evaluated.log_probability)


def test_deterministic_action_is_bounded_and_repeatable() -> None:
    model = _model()
    observation = torch.tensor([[0.3, -0.2]])

    first = model.deterministic_action(observation)
    second = model.deterministic_action(observation)

    torch.testing.assert_close(first, second)
    assert torch.all(first.abs() <= 1.0)
