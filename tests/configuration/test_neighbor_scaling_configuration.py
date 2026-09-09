"""Protocol checks for the controlled swarm-size/neighbor-count suite."""

from pathlib import Path

from uav_swarm_control.communication import neighbor_scaling_grid
from uav_swarm_control.configuration import load_communication_experiment_config
from uav_swarm_control.obstacles import ObstacleScenario

ROOT = Path(__file__).parents[2]
DIRECTORY = ROOT / "configs/experiment/neighbor-scaling"


def test_neighbor_scaling_configs_match_declared_n_by_k_grid() -> None:
    expected = neighbor_scaling_grid((4, 8, 16, 32), (0, 1, 2, 3, 4, 6, 8))
    expected_by_size: dict[int, set[int]] = {}
    for condition in expected:
        expected_by_size.setdefault(condition.swarm_size, set()).add(condition.requested_neighbors)

    loaded = [
        load_communication_experiment_config(DIRECTORY / f"sphere-{size}-uav.yaml")
        for size in (4, 8, 16, 32)
    ]
    assert [item.mappo.experiment.formation.num_agents for item in loaded] == [4, 8, 16, 32]
    for config in loaded:
        size = config.mappo.experiment.formation.num_agents
        clear_global = {
            condition.requested_neighbors
            for condition in config.evaluation_conditions
            if condition.sensing_radius_m is None
            and condition.obstacle_scenario is ObstacleScenario.NONE
        }
        assert clear_global == expected_by_size[size]
        assert config.mappo.experiment.formation.kind.value == "sphere"
        assert config.encoder.payload_bytes_per_neighbor == 24
        assert len(config.training_seeds) == 5
