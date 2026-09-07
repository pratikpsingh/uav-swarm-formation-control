"""Independent numerical and end-to-end checks for reproducible baseline experiments."""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.cli import main
from uav_swarm_control.configuration import ConfigurationError, PyBulletSimulatorConfig
from uav_swarm_control.configuration.baseline import load_baseline_config
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.environments.drone_backend import RigidBodyState, SimulatorMetadataValue
from uav_swarm_control.environments.formation_progress import FormationProgressEnvironment
from uav_swarm_control.environments.pybullet import PyBulletSwarmEnvironment
from uav_swarm_control.evaluation.baseline import smoke_config
from uav_swarm_control.evaluation.benchmark import evaluate_benchmark, summarize_seeds
from uav_swarm_control.formations import FormationKind
from uav_swarm_control.observations import LocalObservations

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiment/baseline/triangle-3-uav.yaml"


class ZeroController:
    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        return NormalizedVelocityActions(
            observations.agent_ids, np.zeros((observations.num_agents, 3), dtype=np.float32)
        )


class TrajectoryBackend:
    """Prescribed trajectories independent of PID/physics for metric arithmetic."""

    def __init__(self, *, collapse: bool = False) -> None:
        self.collapse = collapse
        self.positions = np.zeros((3, 3), dtype=np.float32)
        self.initial = self.positions.copy()
        self.steps = 0
        self.closed = False

    @property
    def metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        return {"simulator": "prescribed-trajectory"}

    def state(self) -> RigidBodyState:
        quaternions = np.zeros((3, 4), dtype=np.float32)
        quaternions[:, 3] = 1
        return RigidBodyState(
            self.positions,
            quaternions,
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )

    def reset(self, initial_positions: Float32Array, *, seed: int) -> RigidBodyState:
        self.initial = initial_positions.copy()
        self.positions = self.initial.copy()
        self.steps = 0
        return self.state()

    def step_velocity(self, target_velocities_mps: Float32Array) -> RigidBodyState:
        self.steps += 1
        if self.collapse:
            self.positions = (
                np.tile(self.initial.mean(axis=0), (3, 1))
                if self.steps == 1
                else self.initial.copy()
            )
        else:
            self.positions[:, 0] += 0.1
        return self.state()

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("count,budget", [(3, 10_000_000), (4, 30_000_000), (5, 60_000_000)])
def test_research_protocol_and_smoke_are_explicit(count: int, budget: int) -> None:
    names = {3: "triangle-3-uav.yaml", 4: "square-4-uav.yaml", 5: "pentagon-5-uav.yaml"}
    config = load_baseline_config(ROOT / "configs/experiment/baseline" / names[count])
    assert config.mappo.algorithm.ppo.total_steps == budget
    assert len(config.training_seeds) == 5
    assert config.mappo.experiment.environment.max_episode_steps == 242
    smoke = smoke_config(config)
    assert smoke.profile == "smoke"
    assert smoke.mappo.algorithm.ppo.total_steps == 256
    assert config.profile == "research"


@pytest.mark.parametrize("seeds", [[1, 2, 3, 4], [1, 2, 3, 4, 4], [True, 2, 3, 4, 5]])
def test_seed_protocol_rejects_invalid_independent_runs(
    tmp_path: Path, seeds: list[object]
) -> None:
    raw = cast(dict[str, object], yaml.safe_load(CONFIG.read_text()))
    protocol = cast(dict[str, object], raw["protocol"])
    protocol["training_seeds"] = seeds
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigurationError):
        load_baseline_config(path)


def test_progress_reward_and_reset_history() -> None:
    config = load_baseline_config(CONFIG).physics
    experiment = replace(
        config.experiment, task=replace(config.experiment.task, initial_position_noise_m=0)
    )
    backend = TrajectoryBackend()

    def factory(
        config: PyBulletSimulatorConfig, count: int, positions: Float32Array
    ) -> TrajectoryBackend:
        return backend

    environment = FormationProgressEnvironment(
        replace(config, experiment=experiment), backend_factory=factory
    )
    try:
        for _ in range(2):
            observations = environment.reset(seed=11).observations
            transition = environment.step(ZeroController().act(observations))
            # Every vehicle progresses 0.1 m; weight=10; rigid translation has no shape cost.
            np.testing.assert_allclose(transition.rewards, np.ones(3), atol=2e-6)
            assert transition.metrics["reward/navigation_mean"] == pytest.approx(1.0, abs=2e-6)
    finally:
        environment.close()


def test_episode_metrics_retain_midflight_collision_and_initial_path_segment() -> None:
    config = load_baseline_config(CONFIG).physics
    experiment = replace(
        config.experiment,
        formation=replace(
            config.experiment.formation, kind=FormationKind.LINE, center_m=(3.0, 0.0, 1.0)
        ),
        environment=replace(config.experiment.environment, max_episode_steps=2),
        task=replace(
            config.experiment.task,
            initial_center_m=(0.0, 0.0, 1.0),
            initial_position_noise_m=0.0,
            terminate_on_collision=False,
        ),
    )
    backend = TrajectoryBackend(collapse=True)

    def backend_factory(
        config: PyBulletSimulatorConfig, count: int, positions: Float32Array
    ) -> TrajectoryBackend:
        return backend

    result = evaluate_benchmark(
        lambda: PyBulletSwarmEnvironment(
            replace(config, experiment=experiment), backend_factory=backend_factory
        ),
        ZeroController(),
        episodes=1,
        seed=99,
        horizon=2,
        time_step_seconds=1 / 30,
        collision_distance_m=0.2,
    )[0]
    assert result["collision_any"] == 1
    assert result["collision_pair_steps"] == 3
    assert result["minimum_clearance_m"] == pytest.approx(-0.2)
    assert result["mean_agent_path_length_m"] == pytest.approx(4 / 3)
    assert result["final_normalized_shape_rmse"] == pytest.approx(0)
    assert result["mean_normalized_shape_rmse"] > 0
    assert isinstance(result["episode_seed"], int)
    assert backend.closed


def test_uncertainty_uses_training_seed_means() -> None:
    result = summarize_seeds([{"success": 0.0}, {"success": 1.0}])
    assert result["seed_count"] == 2
    assert result["metrics"] == {"success": {"mean": 0.5, "sample_std": pytest.approx(2**-0.5)}}


def test_cli_smoke_checkpoints_and_completed_seed_resume(tmp_path: Path) -> None:
    arguments = [
        "run-baseline",
        "--config",
        str(CONFIG),
        "--smoke",
        "--output",
        str(tmp_path),
        "--project-root",
        str(ROOT),
    ]
    main(arguments)
    directory = tmp_path / "smoke/baseline-triangle-3-uav"
    result = json.loads((directory / "seed-11/result.json").read_text())
    assert result["summary"]["training_environment_steps"] == 256
    assert result["summary"]["training_agent_samples"] == 768
    assert [row["episode_seed"] for row in result["episodes"]] == [
        row["episode_seed"] for row in result["scripted_episodes"]
    ]
    assert json.loads((directory / "summary.json").read_text())["seed_count"] == 5
    original = (directory / "seed-11/model.pt").stat().st_mtime_ns
    main([*arguments, "--resume"])
    assert (directory / "seed-11/model.pt").stat().st_mtime_ns == original
    evaluation_path = tmp_path / "reevaluated.json"
    main(
        [
            "evaluate-baseline",
            "--config",
            str(CONFIG),
            "--smoke",
            "--checkpoint",
            str(directory / "seed-11/model.pt"),
            "--output",
            str(evaluation_path),
            "--project-root",
            str(ROOT),
        ]
    )
    reevaluated = json.loads(evaluation_path.read_text())
    assert reevaluated["episodes"] == result["episodes"]
    with pytest.raises(SystemExit):
        main(arguments)
    (directory / "seed-11/model.pt").write_bytes(b"invalid-checkpoint")
    with pytest.raises(SystemExit):
        main([*arguments, "--resume"])
