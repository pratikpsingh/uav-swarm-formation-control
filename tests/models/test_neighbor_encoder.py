"""Behavior tests for masked permutation-invariant neighbor aggregation."""

import torch

from uav_swarm_control.models import NeighborEncoderSpec, SharedActorCentralCritic


def _model() -> SharedActorCentralCritic:
    torch.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    return SharedActorCentralCritic(
        local_observation_size=35,
        centralized_state_size=12,
        action_size=3,
        num_agents=4,
        actor_hidden_sizes=(16, 16),
        critic_hidden_sizes=(16, 16),
        initial_log_standard_deviation=-0.5,
        neighbor_encoder=NeighborEncoderSpec(6, 3, 6, 8, (12,)),
    )


def test_neighbor_slot_permutation_does_not_change_action() -> None:
    model = _model()
    observation = torch.randn(2, 35)
    observation[:, 24:27] = torch.tensor([1.0, 0.0, 1.0])
    permutation = torch.tensor([2, 0, 1])
    permuted = observation.clone()
    neighbor_rows = observation[:, 6:24].reshape(2, 3, 6)
    permuted[:, 6:24] = neighbor_rows[:, permutation].reshape(2, 18)
    permuted[:, 24:27] = observation[:, 24:27][:, permutation]

    torch.testing.assert_close(
        model.deterministic_actions(permuted),
        model.deterministic_actions(observation),
    )


def test_masked_neighbor_values_do_not_change_action() -> None:
    model = _model()
    first = torch.randn(2, 35)
    first[:, 24:27] = 0.0
    second = first.clone()
    second[:, 6:24] = torch.randn(2, 18) * 1000.0

    torch.testing.assert_close(
        model.deterministic_actions(first),
        model.deterministic_actions(second),
    )
