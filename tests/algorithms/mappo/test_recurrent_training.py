"""End-to-end tests for sequence-correct recurrent MAPPO."""

import math
from dataclasses import replace
from pathlib import Path

from uav_swarm_control.algorithms.mappo import (
    evaluate_recurrent_mappo,
    train_recurrent_mappo,
)
from uav_swarm_control.configuration import (
    RecurrentMAPPOConfig,
    RecurrentObservationProfile,
    load_mappo_experiment_config,
)
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment

CONFIG = Path(__file__).parents[3] / "configs/experiment/mappo_triangle_kinematic.yaml"


def test_recurrent_mappo_trains_and_evaluates_actor_only() -> None:
    base = load_mappo_experiment_config(CONFIG)
    algorithm = replace(
        base.algorithm,
        ppo=replace(
            base.algorithm.ppo,
            total_steps=16,
            rollout_steps=8,
            minibatch_size=8,
            update_epochs=1,
            device="cpu",
        ),
        num_environments=1,
    )
    config = RecurrentMAPPOConfig(
        mappo=algorithm,
        feature_size=8,
        hidden_size=8,
        num_layers=1,
        sequence_length=4,
        sequences_per_minibatch=1,
        observation_profile=RecurrentObservationProfile.MASKED_SET,
    )

    def factory() -> KinematicSwarmEnvironment:
        return KinematicSwarmEnvironment(base.experiment)

    result = train_recurrent_mappo(factory, config, seed=7)

    assert result.environment_steps == 16
    assert result.agent_samples == 48
    assert len(result.history) == 2
    assert all(
        math.isfinite(value)
        for update in result.history
        for value in (
            update.policy_loss,
            update.value_loss,
            update.entropy,
            update.approximate_kl,
        )
    )

    evaluation = evaluate_recurrent_mappo(
        result.model,
        factory,
        episodes=1,
        seed=17,
        observation_profile=config.observation_profile,
    )
    assert evaluation.episodes == 1
    assert math.isfinite(evaluation.mean_episode_return)
