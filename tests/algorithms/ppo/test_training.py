"""End-to-end PPO learning tests on the isolated reference task."""

import pytest

from uav_swarm_control.algorithms.ppo import evaluate_continuous_bandit, train_ppo
from uav_swarm_control.configuration import ContinuousBanditConfig, PPOConfig
from uav_swarm_control.environments import ContinuousTargetBandit


def _training_config() -> PPOConfig:
    return PPOConfig(
        total_steps=2048,
        rollout_steps=256,
        update_epochs=8,
        minibatch_size=64,
        learning_rate=0.003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coefficient=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_gradient_norm=0.5,
        hidden_sizes=(32, 32),
        initial_log_standard_deviation=-0.5,
        device="cpu",
    )


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_ppo_learns_continuous_target_across_fixed_seeds(seed: int) -> None:
    task = ContinuousBanditConfig(-0.8, 0.8, 0.1, 256)

    result = train_ppo(ContinuousTargetBandit(task), _training_config(), seed=seed)
    evaluation = evaluate_continuous_bandit(
        result.model,
        task,
        seed=seed + 1000,
        device=result.device,
    )

    assert result.environment_steps == 2048
    assert len(result.history) == 8
    assert evaluation.action_mse < 0.01
