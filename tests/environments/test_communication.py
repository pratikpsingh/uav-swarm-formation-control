"""Integration tests for masked communication conditions over a fake backend."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.communication import (
    CommunicationCondition,
    CommunicationRegimen,
    CommunicationRegimenKind,
)
from uav_swarm_control.configuration import (
    PyBulletSimulatorConfig,
    load_communication_experiment_config,
)
from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.environments.drone_backend import RigidBodyState, SimulatorMetadataValue
from uav_swarm_control.obstacles import ObstacleScenario

CONFIG = Path(__file__).parents[2] / "configs/experiment/stage11_plane_4uav.yaml"


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


def test_condition_changes_mask_without_changing_tensor_capacity() -> None:
    config = load_communication_experiment_config(CONFIG)
    condition = CommunicationCondition(
        "one-global-clear",
        1,
        None,
        ObstacleScenario.NONE,
    )
    regimen = CommunicationRegimen(
        "fixed-one",
        CommunicationRegimenKind.FIXED,
        (condition,),
    )
    environment = CommunicationFormationEnvironment(
        config.physics,
        config.pose,
        config.variant,
        config.field,
        regimen,
        payload_bytes_per_neighbor=24,
        backend_factory=_backend,
    )
    try:
        reset = environment.reset(seed=17)
        actions = NormalizedVelocityActions(
            environment.agent_ids,
            np.zeros((4, 3), dtype=np.float32),
        )
        transition = environment.step(actions)

        assert reset.observations.neighbors.shape == (4, 3, 6)
        assert reset.observations.neighbor_mask.sum(axis=1).tolist() == [1, 1, 1, 1]
        assert transition.metrics["communication/actual_degree_mean"] == 1.0
        assert transition.metrics["communication/received_payload_bytes"] == 96.0
        assert environment.episode_metadata["requested_neighbors"] == 1.0
    finally:
        environment.close()


def test_variable_condition_selection_is_repeatable_from_reset_seed() -> None:
    config = load_communication_experiment_config(CONFIG)
    regimen = config.training_regimens[-1]
    first = CommunicationFormationEnvironment(
        config.physics,
        config.pose,
        config.variant,
        config.field,
        regimen,
        payload_bytes_per_neighbor=24,
        training_steps_per_environment=10,
        backend_factory=_backend,
    )
    second = CommunicationFormationEnvironment(
        config.physics,
        config.pose,
        config.variant,
        config.field,
        regimen,
        payload_bytes_per_neighbor=24,
        training_steps_per_environment=10,
        backend_factory=_backend,
    )
    try:
        first.reset(seed=29)
        second.reset(seed=29)
        assert first.communication_condition == second.communication_condition
    finally:
        first.close()
        second.close()
