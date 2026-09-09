"""Behavior tests for construction and obstacle-triggered morphing environments."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.agents import AgentId
from uav_swarm_control.communication import CommunicationRegimen, CommunicationRegimenKind
from uav_swarm_control.configuration import (
    PyBulletSimulatorConfig,
    load_recurrent_study_config,
)
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.environments.drone_backend import RigidBodyState, SimulatorMetadataValue
from uav_swarm_control.environments.morphing import MorphingFormationEnvironment
from uav_swarm_control.environments.research import MissionFormationEnvironment
from uav_swarm_control.missions import MissionPhase, MorphingPhase
from uav_swarm_control.obstacles import ObstacleField

CONFIG = Path(__file__).parents[2] / "configs/experiment/recurrent-study/sphere-8-uav.yaml"


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


def _regimen(name: str) -> CommunicationRegimen:
    config = load_recurrent_study_config(CONFIG)
    condition = next(
        item for item in config.communication.evaluation_conditions if item.name == name
    )
    return CommunicationRegimen("test-condition", CommunicationRegimenKind.FIXED, (condition,))


def _zero_actions(agent_ids: Sequence[AgentId]) -> NormalizedVelocityActions:
    return NormalizedVelocityActions(
        agent_ids,
        np.zeros((len(agent_ids), 3), dtype=np.float32),
    )


def test_construction_reset_places_uavs_near_declared_ground_altitude() -> None:
    config = load_recurrent_study_config(CONFIG)
    mission = config.mission
    environment = MissionFormationEnvironment(
        config.communication.physics,
        config.communication.pose,
        config.communication.variant,
        config.communication.field,
        _regimen("kall-global-clear"),
        formation_kinds=config.formation_kinds,
        payload_bytes_per_neighbor=config.communication.encoder.payload_bytes_per_neighbor,
        initial_ground_altitude_m=mission.initial_ground_altitude_m,
        construction_altitude_m=mission.construction_altitude_m,
        construction_hold_steps=mission.construction_hold_steps,
        waypoint_spacing_m=mission.waypoint_spacing_m,
        waypoint_hold_steps=mission.waypoint_hold_steps,
        backend_factory=_backend,
    )
    try:
        environment.reset(seed=13)

        noise = config.communication.mappo.experiment.task.initial_position_noise_m
        assert np.all(
            np.abs(environment.positions[:, 2] - mission.initial_ground_altitude_m) <= noise + 1e-6
        )
        assert environment.target_positions[:, 2].mean() == pytest.approx(
            mission.construction_altitude_m
        )
        assert environment.mission_phase is MissionPhase.CONSTRUCTING
    finally:
        environment.close()


def test_obstacle_in_corridor_starts_rate_limited_morph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_recurrent_study_config(CONFIG)
    morphing = config.morphing
    environment = MorphingFormationEnvironment(
        config.communication.physics,
        config.communication.pose,
        config.communication.variant,
        config.communication.field,
        _regimen("kall-global-mixed"),
        formation_kinds=config.formation_kinds,
        payload_bytes_per_neighbor=config.communication.encoder.payload_bytes_per_neighbor,
        morph_rate_per_second=morphing.rate_per_second,
        release_hold_steps=morphing.release_hold_steps,
        trigger_clearance_m=morphing.trigger_clearance_m,
        backend_factory=_backend,
    )

    def blocking_field(*args: object, **kwargs: object) -> ObstacleField:
        positions = np.asarray(args[1], dtype=np.float64)
        targets = np.asarray(args[2], dtype=np.float64)
        center = (positions.mean(axis=0) + targets.mean(axis=0)) / 2.0
        return ObstacleField(center.reshape(1, 3), np.zeros((1, 3)), np.array([0.15]))

    monkeypatch.setattr(
        "uav_swarm_control.environments.obstacles.sample_obstacle_field",
        blocking_field,
    )
    try:
        environment.reset(seed=19)
        transition = environment.step(_zero_actions(environment.agent_ids))

        assert environment.morphing_phase is MorphingPhase.DEFORMING
        assert 0.0 < transition.metrics["morph_alpha"] < 1.0
        assert transition.metrics["corridor_blocked"] == 1.0
        assert transition.metrics["deformation_steps"] == 1.0
    finally:
        environment.close()
