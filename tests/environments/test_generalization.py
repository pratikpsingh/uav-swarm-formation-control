"""Contract tests for episode-randomized three-dimensional tasks."""

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.configuration import (
    AssignmentMode,
    CoordinateFrame,
    GeneralizationVariant,
    PyBulletSimulatorConfig,
    load_generalization_config,
)
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.environments.drone_backend import (
    RigidBodyState,
    SimulatorMetadataValue,
)
from uav_swarm_control.environments.generalization import GeneralizedFormationEnvironment
from uav_swarm_control.evaluation.benchmark import evaluate_benchmark
from uav_swarm_control.formations import FormationPoseRange, rotation_matrix_from_euler
from uav_swarm_control.observations import LocalObservations

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiment/stage9_plane_4uav.yaml"


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


class RecordingBackend:
    def __init__(self, control_frequency_hz: int) -> None:
        self.control_frequency_hz = control_frequency_hz
        self.positions: Float32Array | None = None
        self.initial_positions: Float32Array | None = None
        self.last_velocities: Float32Array | None = None

    @property
    def metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        return {"simulator": "recording-double"}

    def reset(self, initial_positions: Float32Array, *, seed: int) -> RigidBodyState:
        self.initial_positions = initial_positions.copy()
        self.positions = initial_positions.copy()
        return _state(self.positions)

    def step_velocity(self, target_velocities_mps: Float32Array) -> RigidBodyState:
        if self.positions is None:
            raise RuntimeError("reset required")
        self.last_velocities = target_velocities_mps.copy()
        self.positions += target_velocities_mps / self.control_frequency_hz
        return _state(self.positions, target_velocities_mps)

    def close(self) -> None:
        pass


class RecordingFactory:
    def __init__(self) -> None:
        self.backend: RecordingBackend | None = None

    def __call__(
        self,
        config: PyBulletSimulatorConfig,
        num_drones: int,
        initial_positions: Float32Array,
    ) -> RecordingBackend:
        assert initial_positions.shape == (num_drones, 3)
        self.backend = RecordingBackend(config.control_frequency_hz)
        return self.backend


class FixedTargetActionController:
    def act(self, observations: LocalObservations) -> NormalizedVelocityActions:
        actions = np.full((observations.num_agents, 3), 1.0, dtype=np.float32)
        return NormalizedVelocityActions(observations.agent_ids, actions)


def _variant(assignment: AssignmentMode, frame: CoordinateFrame) -> GeneralizationVariant:
    return GeneralizationVariant(f"{assignment.value}-{frame.value}", assignment, frame)


def test_pose_randomization_is_seeded_but_initial_geometry_is_independent() -> None:
    config = load_generalization_config(CONFIG)
    fixed_factory = RecordingFactory()
    minimum_factory = RecordingFactory()
    fixed = GeneralizedFormationEnvironment(
        config.physics,
        config.training_pose,
        _variant(AssignmentMode.FIXED, CoordinateFrame.WORLD),
        backend_factory=fixed_factory,
    )
    minimum = GeneralizedFormationEnvironment(
        config.physics,
        config.training_pose,
        _variant(AssignmentMode.MINIMUM_DISTANCE, CoordinateFrame.WORLD),
        backend_factory=minimum_factory,
    )
    try:
        fixed.reset(seed=71)
        minimum.reset(seed=71)

        assert fixed_factory.backend is not None
        assert minimum_factory.backend is not None
        np.testing.assert_array_equal(
            fixed_factory.backend.initial_positions,
            minimum_factory.backend.initial_positions,
        )
        assert fixed.episode_metadata["target_scale"] == minimum.episode_metadata["target_scale"]
        assert (
            minimum.episode_metadata["nominal_assignment_total_distance_m"]
            <= fixed.episode_metadata["nominal_assignment_total_distance_m"]
        )
        first_targets = minimum.target_positions.copy()
        minimum.reset(seed=71)
        np.testing.assert_array_equal(minimum.target_positions, first_targets)
        minimum.reset(seed=72)
        assert not np.array_equal(minimum.target_positions, first_targets)
    finally:
        fixed.close()
        minimum.close()


def test_target_frame_rotates_observations_critic_and_actions() -> None:
    config = load_generalization_config(CONFIG)
    exact_pose = FormationPoseRange(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, math.pi / 2),
        (0.0, 0.0, math.pi / 2),
        1.0,
        1.0,
    )
    factory = RecordingFactory()
    environment = GeneralizedFormationEnvironment(
        config.physics,
        exact_pose,
        _variant(AssignmentMode.FIXED, CoordinateFrame.TARGET),
        backend_factory=factory,
    )
    try:
        reset = environment.reset(seed=13)
        rotation = rotation_matrix_from_euler(0.0, 0.0, math.pi / 2)
        expected_displacement = (environment.target_positions - environment.positions) @ rotation
        np.testing.assert_allclose(reset.observations.ego[:, :3], expected_displacement, atol=1e-6)

        count = len(environment.agent_ids)
        local_actions = np.zeros((count, 3), dtype=np.float32)
        local_actions[:, 0] = 1.0
        transition = environment.step(
            NormalizedVelocityActions(environment.agent_ids, local_actions)
        )

        assert factory.backend is not None
        assert factory.backend.last_velocities is not None
        expected_velocity = np.tile([0.0, 0.5, 0.0], (count, 1))
        np.testing.assert_allclose(factory.backend.last_velocities, expected_velocity, atol=1e-6)
        assert transition.metrics["action_frame_clip_fraction"] == 0.0
        assert reset.centralized_state.size == transition.centralized_state.size
    finally:
        environment.close()


def test_benchmark_persists_mean_target_frame_action_clipping() -> None:
    config = load_generalization_config(CONFIG)
    exact_pose = FormationPoseRange(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, math.pi / 4),
        (0.0, 0.0, math.pi / 4),
        1.0,
        1.0,
    )

    def factory() -> GeneralizedFormationEnvironment:
        return GeneralizedFormationEnvironment(
            config.physics,
            exact_pose,
            _variant(AssignmentMode.FIXED, CoordinateFrame.TARGET),
            backend_factory=RecordingFactory(),
        )

    record = evaluate_benchmark(
        factory,
        FixedTargetActionController(),
        episodes=1,
        seed=9,
        horizon=config.mappo.experiment.environment.max_episode_steps,
        time_step_seconds=config.mappo.experiment.environment.time_step_seconds,
        collision_distance_m=config.mappo.experiment.task.collision_distance_m,
    )[0]

    assert record["action_frame_clip_fraction"] > 0.0
