"""Common swarm task backed by Crazyflie rigid-body simulation."""

import math
from collections.abc import Callable, Mapping

import numpy as np

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.agents import AgentId, sequential_agent_ids
from uav_swarm_control.configuration import PyBulletExperimentConfig, PyBulletSimulatorConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.drone_backend import (
    DronePhysicsBackend,
    RigidBodyState,
    SimulatorMetadataValue,
)
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.evaluation.kinematics import evaluate_kinematic_state
from uav_swarm_control.formations import create_formation, rotation_matrix_from_euler, transform
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.observations import (
    CentralizedState,
    LocalObservations,
    build_centralized_state,
    build_local_observations,
)
from uav_swarm_control.rewards.components import compute_reward
from uav_swarm_control.seeding import RandomStream, make_rng

type BackendFactory = Callable[
    [PyBulletSimulatorConfig, int, Float32Array],
    DronePhysicsBackend,
]


class PyBulletSwarmEnvironment:
    """Run the common formation task using a velocity-controlled physics backend."""

    def __init__(
        self,
        config: PyBulletExperimentConfig,
        *,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        self._config = config
        experiment = config.experiment
        self._agent_ids = sequential_agent_ids(experiment.formation.num_agents)
        template = create_formation(
            experiment.formation.kind,
            experiment.formation.num_agents,
            experiment.formation.spacing_m,
        )
        roll, pitch, yaw = experiment.formation.euler_radians
        rotation = rotation_matrix_from_euler(roll, pitch, yaw)
        self._relative_targets = template @ rotation.T
        self._targets = transform(template, rotation=rotation, offset=experiment.formation.center_m)
        nominal_initial = self._relative_targets + np.asarray(
            experiment.task.initial_center_m,
            dtype=np.float64,
        )
        self._backend = backend_factory(
            config.simulator,
            len(self._agent_ids),
            nominal_initial.astype(np.float32),
        )
        self._rigid_state: RigidBodyState | None = None
        self._previous_actions: FloatArray | None = None
        self._step_count = 0
        self._success_streak = 0
        self._episode_done = False
        self._closed = False

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        """Stable identifiers in simulator row order."""
        return self._agent_ids

    @property
    def target_positions(self) -> FloatArray:
        """Return an immutable copy of assigned world targets."""
        return self._immutable_float64(self._targets)

    @property
    def positions(self) -> FloatArray:
        """Return an immutable copy of current simulated positions."""
        state, _ = self._state()
        return self._immutable_float64(state.positions)

    @property
    def simulator_metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        """Return backend provenance and resolved physics timing."""
        return self._backend.metadata

    def reset(self, *, seed: int) -> ResetResult:
        """Reset simulator and PID memory from a deterministic position sample."""
        if self._closed:
            raise RuntimeError("cannot reset a closed environment.")
        rng = make_rng(seed, RandomStream.INITIAL_STATE)
        noise_limit = self._config.experiment.task.initial_position_noise_m
        noise = rng.uniform(-noise_limit, noise_limit, size=(len(self._agent_ids), 3))
        initial_center = np.asarray(
            self._config.experiment.task.initial_center_m,
            dtype=np.float64,
        )
        initial_positions = self._relative_targets + initial_center + noise
        if np.any(initial_positions[:, 2] <= 0.0):
            raise ValueError("all sampled initial drone altitudes must be greater than zero.")
        self._rigid_state = self._backend.reset(
            initial_positions.astype(np.float32),
            seed=seed,
        )
        self._validate_drone_count(self._rigid_state)
        self._previous_actions = np.zeros((len(self._agent_ids), 3), dtype=np.float64)
        self._step_count = 0
        self._success_streak = 0
        self._episode_done = False
        return ResetResult(self._local_observations(), self._centralized_state())

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        """Convert normalized actions to m/s, run PID, and advance rigid-body physics."""
        _, previous_actions = self._state()
        if self._episode_done:
            raise RuntimeError("episode is complete; call reset before step.")
        if tuple(actions.agent_ids) != self._agent_ids:
            raise ValueError("action agent_ids must exactly match environment row order.")

        normalized_actions = np.asarray(actions.values, dtype=np.float64)
        desired_velocities = (
            normalized_actions * self._config.experiment.environment.max_velocity_component_mps
        )
        state = self._backend.step_velocity(desired_velocities.astype(np.float32))
        self._validate_drone_count(state)
        self._rigid_state = state
        self._step_count += 1

        positions = np.asarray(state.positions, dtype=np.float64)
        task = self._config.experiment.task
        state_metrics = evaluate_kinematic_state(
            positions,
            self._targets,
            normalized_actions,
            previous_actions,
            collision_distance_m=task.collision_distance_m,
            success_tolerance_m=task.success_tolerance_m,
        )
        self._success_streak = self._success_streak + 1 if state_metrics.within_tolerance else 0
        success = self._success_streak >= task.success_hold_steps
        collision_failure = task.terminate_on_collision and state_metrics.collision_pairs > 0
        terminated = success or collision_failure
        truncated = (
            not terminated
            and self._step_count >= self._config.experiment.environment.max_episode_steps
        )

        reward = compute_reward(
            positions,
            self._targets,
            normalized_actions,
            previous_actions,
            collision_distance_m=task.collision_distance_m,
            success=success,
            config=self._config.experiment.reward,
        )
        metrics = state_metrics.to_dict()
        metrics.update(reward.mean_metrics())
        metrics.update(self._rigid_body_metrics(state))
        metrics.update(
            {
                "success": float(success),
                "collision_failure": float(collision_failure),
                "step_count": float(self._step_count),
                "success_streak": float(self._success_streak),
                "simulation_time_seconds": (
                    self._step_count * self._config.experiment.environment.time_step_seconds
                ),
                "physics_step_count": float(
                    self._step_count * self._config.simulator.physics_steps_per_control
                ),
            }
        )
        self._previous_actions = normalized_actions.copy()
        self._episode_done = terminated or truncated
        return StepResult(
            observations=self._local_observations(),
            centralized_state=self._centralized_state(),
            rewards=reward.total,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        """Release the simulator connection; repeated calls are safe."""
        if not self._closed:
            self._backend.close()
            self._closed = True

    def _state(self) -> tuple[RigidBodyState, FloatArray]:
        if self._rigid_state is None or self._previous_actions is None:
            raise RuntimeError("environment must be reset before use.")
        return self._rigid_state, self._previous_actions

    def _local_observations(self) -> LocalObservations:
        state, _ = self._state()
        return build_local_observations(
            self._agent_ids,
            state.positions,
            state.linear_velocities_mps,
            self._targets,
            max_neighbors=self._config.experiment.observation.max_neighbors,
            neighbor_radius_m=self._config.experiment.observation.neighbor_radius_m,
        )

    def _centralized_state(self) -> CentralizedState:
        state, _ = self._state()
        return build_centralized_state(
            state.positions,
            state.linear_velocities_mps,
            self._targets,
        )

    def _validate_drone_count(self, state: RigidBodyState) -> None:
        if state.num_drones != len(self._agent_ids):
            raise RuntimeError(
                f"backend returned {state.num_drones} drones; expected {len(self._agent_ids)}."
            )

    @staticmethod
    def _rigid_body_metrics(state: RigidBodyState) -> dict[str, float]:
        attitudes = np.asarray(state.euler_angles_radians, dtype=np.float64)
        angular_velocities = np.asarray(state.angular_velocities_rad_s, dtype=np.float64)
        motor_rpms = np.asarray(state.motor_rpms, dtype=np.float64)
        return {
            "attitude_rms_rad": math.sqrt(float(np.mean(np.square(attitudes)))),
            "angular_velocity_rms_rad_s": math.sqrt(float(np.mean(np.square(angular_velocities)))),
            "motor_rpm_mean": float(np.mean(motor_rpms)),
            "motor_rpm_max": float(np.max(motor_rpms)),
        }

    @staticmethod
    def _immutable_float64(values: object) -> FloatArray:
        result = np.asarray(values, dtype=np.float64).copy()
        result.setflags(write=False)
        return result
