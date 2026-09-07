"""Behavior tests for oracle obstacles over a deterministic flight backend."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.configuration import (
    CurriculumPhase,
    PyBulletSimulatorConfig,
    load_obstacle_experiment_config,
)
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.environments.drone_backend import RigidBodyState, SimulatorMetadataValue
from uav_swarm_control.environments.obstacles import ObstacleFormationEnvironment
from uav_swarm_control.obstacles import ObstacleField, ObstacleScenario

CONFIG = Path(__file__).parents[2] / "configs/experiment/obstacle-avoidance/plane-4-uav.yaml"


def _state(positions: Float32Array, velocities: Float32Array | None = None) -> RigidBodyState:
    count = positions.shape[0]
    quaternions = np.zeros((count, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    return RigidBodyState(
        positions,
        quaternions,
        np.zeros((count, 3), dtype=np.float32),
        np.zeros((count, 3), dtype=np.float32) if velocities is None else velocities,
        np.zeros((count, 3), dtype=np.float32),
        np.zeros((count, 4), dtype=np.float32),
    )


class DeterministicBackend:
    def __init__(self, frequency: int) -> None:
        self.frequency = frequency
        self.positions: Float32Array | None = None

    @property
    def metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        return {"simulator": "deterministic-double"}

    def reset(self, initial_positions: Float32Array, *, seed: int) -> RigidBodyState:
        self.positions = initial_positions.copy()
        return _state(self.positions)

    def step_velocity(self, target_velocities_mps: Float32Array) -> RigidBodyState:
        if self.positions is None:
            raise RuntimeError("reset required")
        self.positions += target_velocities_mps / self.frequency
        return _state(self.positions, target_velocities_mps)

    def close(self) -> None:
        pass


def _backend(
    config: PyBulletSimulatorConfig, num_drones: int, initial_positions: Float32Array
) -> DeterministicBackend:
    assert initial_positions.shape == (num_drones, 3)
    return DeterministicBackend(config.control_frequency_hz)


def _environment(
    scenario: ObstacleScenario,
    *,
    training_steps: int | None = None,
    phases: tuple[CurriculumPhase, ...] | None = None,
) -> ObstacleFormationEnvironment:
    config = load_obstacle_experiment_config(CONFIG)
    selected = phases or (CurriculumPhase(1.0, scenario),)
    return ObstacleFormationEnvironment(
        config.physics,
        config.pose,
        config.variant,
        config.field,
        selected,
        training_steps_per_environment=training_steps,
        backend_factory=_backend,
    )


def _zero_actions(environment: ObstacleFormationEnvironment) -> NormalizedVelocityActions:
    return NormalizedVelocityActions(
        environment.agent_ids, np.zeros((len(environment.agent_ids), 3), dtype=np.float32)
    )


def test_masked_observations_and_critic_width_do_not_depend_on_active_count() -> None:
    none = _environment(ObstacleScenario.NONE)
    static = _environment(ObstacleScenario.STATIC)
    try:
        none_reset = none.reset(seed=17)
        static_reset = static.reset(seed=17)

        assert none_reset.observations.obstacles is not None
        assert none_reset.observations.obstacle_mask is not None
        assert static_reset.observations.obstacles is not None
        assert static_reset.observations.obstacle_mask is not None
        assert none_reset.observations.obstacles.shape == (4, 3, 7)
        assert not none_reset.observations.obstacle_mask.any()
        assert static_reset.observations.obstacle_mask.sum(axis=1).tolist() == [2, 2, 2, 2]
        assert none_reset.centralized_state.size == static_reset.centralized_state.size
        assert np.all(static_reset.observations.obstacles[:, :2, 6] > 0.0)
    finally:
        none.close()
        static.close()


def test_dynamic_obstacles_advance_at_configured_velocity() -> None:
    environment = _environment(ObstacleScenario.SLOW_DYNAMIC)
    try:
        environment.reset(seed=23)
        before = environment.obstacle_field
        transition = environment.step(_zero_actions(environment))
        after = environment.obstacle_field
        dt = load_obstacle_experiment_config(CONFIG).mappo.experiment.environment.time_step_seconds

        np.testing.assert_allclose(
            after.positions_m, before.positions_m + before.velocities_mps * dt
        )
        assert transition.metrics["dynamic_obstacle_count"] == 2.0
        assert "minimum_obstacle_clearance_m" in transition.metrics
    finally:
        environment.close()


def test_obstacle_overlap_penalizes_and_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _environment(ObstacleScenario.STATIC)

    def colliding_field(*args: object, **kwargs: object) -> ObstacleField:
        position = environment.positions[0:1]
        return ObstacleField(position, np.zeros((1, 3)), np.array([0.2]))

    monkeypatch.setattr(
        "uav_swarm_control.environments.obstacles.sample_obstacle_field", colliding_field
    )
    try:
        environment.reset(seed=29)
        transition = environment.step(_zero_actions(environment))

        assert transition.terminated
        assert not transition.truncated
        assert transition.metrics["obstacle_collision_failure"] == 1.0
        assert transition.metrics["success"] == 0.0
        assert transition.metrics["reward/obstacle_collision_mean"] < 0.0
    finally:
        environment.close()


def test_curriculum_uses_elapsed_environment_steps_at_reset() -> None:
    config = load_obstacle_experiment_config(CONFIG)
    curriculum = config.training_regimens[-1]
    environment = _environment(
        ObstacleScenario.NONE,
        training_steps=2,
        phases=curriculum.phases,
    )
    try:
        environment.reset(seed=31)
        assert environment.obstacle_scenario is ObstacleScenario.NONE
        environment.step(_zero_actions(environment))
        environment.reset(seed=32)
        assert environment.obstacle_scenario is ObstacleScenario.SLOW_DYNAMIC
    finally:
        environment.close()
