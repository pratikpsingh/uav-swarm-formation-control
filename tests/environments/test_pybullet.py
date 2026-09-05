"""Contract tests for the common swarm task over a rigid-body backend."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.configuration import PyBulletSimulatorConfig, load_pybullet_experiment_config
from uav_swarm_control.controllers import ProportionalPositionController
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.environments.drone_backend import (
    RigidBodyState,
    SimulatorMetadataValue,
    immutable_simulator_metadata,
)
from uav_swarm_control.environments.gym_pybullet_drones import SIMULATOR_REVISION
from uav_swarm_control.environments.pybullet import PyBulletSwarmEnvironment
from uav_swarm_control.evaluation.rollout import run_episode

CONFIG_DIRECTORY = Path(__file__).parents[2] / "configs" / "experiment"


def _state(positions: Float32Array, velocities: Float32Array | None = None) -> RigidBodyState:
    count = positions.shape[0]
    linear_velocities = np.zeros((count, 3), dtype=np.float32)
    if velocities is not None:
        linear_velocities = velocities
    quaternions = np.zeros((count, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    return RigidBodyState(
        positions=positions,
        quaternions_xyzw=quaternions,
        euler_angles_radians=np.zeros((count, 3), dtype=np.float32),
        linear_velocities_mps=linear_velocities,
        angular_velocities_rad_s=np.zeros((count, 3), dtype=np.float32),
        motor_rpms=np.full((count, 4), 14000.0, dtype=np.float32),
    )


class FakeBackend:
    """Small deterministic physics double that exposes boundary semantics."""

    def __init__(self, control_frequency_hz: int) -> None:
        self._control_frequency_hz = control_frequency_hz
        self._positions: Float32Array | None = None
        self.last_velocities: Float32Array | None = None
        self.seeds: list[int] = []
        self.closed = False

    @property
    def metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        return immutable_simulator_metadata({"simulator_package": "fake", "deterministic": True})

    def reset(self, initial_positions: Float32Array, *, seed: int) -> RigidBodyState:
        self.seeds.append(seed)
        self._positions = initial_positions.copy()
        return _state(self._positions)

    def step_velocity(self, target_velocities_mps: Float32Array) -> RigidBodyState:
        if self._positions is None:
            raise RuntimeError("reset required")
        self.last_velocities = target_velocities_mps.copy()
        self._positions += target_velocities_mps / self._control_frequency_hz
        return _state(self._positions, target_velocities_mps)

    def close(self) -> None:
        self.closed = True


class FakeBackendFactory:
    def __init__(self) -> None:
        self.backend: FakeBackend | None = None

    def __call__(
        self,
        config: PyBulletSimulatorConfig,
        num_drones: int,
        initial_positions: Float32Array,
    ) -> FakeBackend:
        assert initial_positions.shape == (num_drones, 3)
        self.backend = FakeBackend(config.control_frequency_hz)
        return self.backend


def test_normalized_actions_cross_boundary_as_physical_velocities() -> None:
    config = load_pybullet_experiment_config(CONFIG_DIRECTORY / "pybullet_hover.yaml")
    factory = FakeBackendFactory()
    environment = PyBulletSwarmEnvironment(config, backend_factory=factory)
    reset = environment.reset(seed=17)
    transition = environment.step(
        NormalizedVelocityActions(environment.agent_ids, np.array([[1.0, 0.0, 0.0]]))
    )

    assert factory.backend is not None
    last_velocities = factory.backend.last_velocities
    assert last_velocities is not None
    np.testing.assert_allclose(last_velocities, [[0.5, 0.0, 0.0]])
    assert transition.observations.ego[0, 3] == pytest.approx(0.5)
    assert transition.metrics["physics_step_count"] == 5.0
    assert transition.metrics["simulation_time_seconds"] == pytest.approx(1.0 / 48.0)
    assert reset.centralized_state.values.shape == transition.centralized_state.values.shape


def test_reset_is_seeded_and_close_is_idempotent() -> None:
    config = load_pybullet_experiment_config(CONFIG_DIRECTORY / "pybullet_triangle.yaml")
    factory = FakeBackendFactory()
    environment = PyBulletSwarmEnvironment(config, backend_factory=factory)

    first = environment.reset(seed=91).centralized_state.values.copy()
    second = environment.reset(seed=91).centralized_state.values.copy()
    np.testing.assert_array_equal(first, second)
    environment.close()
    environment.close()

    assert factory.backend is not None
    assert factory.backend.seeds == [91, 91]
    assert factory.backend.closed


@pytest.mark.parametrize("configuration_name", ["pybullet_hover.yaml", "pybullet_triangle.yaml"])
def test_real_scripted_pybullet_completion_gates(configuration_name: str) -> None:
    config = load_pybullet_experiment_config(CONFIG_DIRECTORY / configuration_name)
    environment = PyBulletSwarmEnvironment(config)
    controller = ProportionalPositionController(
        gain_per_second=config.experiment.controller.gain_per_second,
        max_velocity_component_mps=config.experiment.environment.max_velocity_component_mps,
    )
    try:
        result = run_episode(
            environment,
            controller,
            seed=config.experiment.seed,
            safety_step_limit=config.experiment.environment.max_episode_steps,
        )
        metadata = environment.simulator_metadata
    finally:
        environment.close()

    assert result.success
    assert result.final_metrics["collision_pairs"] == 0.0
    assert metadata["simulator_revision"] == SIMULATOR_REVISION
    assert metadata["physics_frequency_hz"] == 240
