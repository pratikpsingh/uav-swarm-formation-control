"""End-to-end MAPPO learning tests on the three-agent kinematic task."""

from dataclasses import replace
from pathlib import Path

import pytest

from uav_swarm_control.algorithms.mappo import evaluate_mappo, train_mappo
from uav_swarm_control.configuration import load_mappo_experiment_config
from uav_swarm_control.environments.kinematic import KinematicSwarmEnvironment

REFERENCE_CONFIG = (
    Path(__file__).parents[3] / "configs" / "experiment" / "mappo_triangle_kinematic.yaml"
)


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_mappo_learns_triangle_task_across_fixed_seeds(seed: int) -> None:
    config = load_mappo_experiment_config(REFERENCE_CONFIG)
    ppo = replace(
        config.algorithm.ppo,
        total_steps=16_384,
        rollout_steps=128,
        minibatch_size=128,
    )
    algorithm = replace(config.algorithm, ppo=ppo)

    def environment_factory() -> KinematicSwarmEnvironment:
        return KinematicSwarmEnvironment(config.experiment)

    result = train_mappo(environment_factory, algorithm, seed=seed)
    evaluation = evaluate_mappo(
        result.model,
        environment_factory,
        episodes=10,
        seed=seed,
        device=result.device,
    )

    assert result.environment_steps == 16_384
    assert result.agent_samples == 49_152
    assert len(result.history) == 32
    assert evaluation.success_rate == 1.0
    assert evaluation.collision_episode_rate == 0.0
    assert evaluation.mean_final_position_rmse_m < 0.1
