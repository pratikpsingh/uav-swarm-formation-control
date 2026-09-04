"""Deterministic first-order 3D point-mass swarm environment."""

import math

import numpy as np

from uav_swarm_control.agents import AgentId, sequential_agent_ids
from uav_swarm_control.configuration import ExperimentConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.evaluation.kinematics import evaluate_kinematic_state
from uav_swarm_control.formations import (
    create_formation,
    rotation_matrix_from_euler,
    transform,
)
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.observations import CentralizedState, LocalObservations
from uav_swarm_control.rewards.components import compute_reward
from uav_swarm_control.seeding import RandomStream, make_rng


class KinematicSwarmEnvironment:
    """A lightweight velocity-controlled swarm with no rigid-body physics."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._agent_ids = sequential_agent_ids(config.formation.num_agents)
        template = create_formation(
            config.formation.kind,
            config.formation.num_agents,
            config.formation.spacing_m,
        )
        roll, pitch, yaw = config.formation.euler_radians
        rotation = rotation_matrix_from_euler(roll, pitch, yaw)
        self._relative_targets = template @ rotation.T
        self._targets = transform(
            template,
            rotation=rotation,
            offset=config.formation.center_m,
        )
        self._positions: FloatArray | None = None
        self._velocities: FloatArray | None = None
        self._previous_actions: FloatArray | None = None
        self._step_count = 0
        self._success_streak = 0
        self._episode_done = False
        self._closed = False

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        """Stable identifiers in environment array order."""
        return self._agent_ids

    @property
    def target_positions(self) -> FloatArray:
        """Return an immutable copy of assigned world targets."""
        result = self._targets.copy()
        result.setflags(write=False)
        return result

    @property
    def positions(self) -> FloatArray:
        """Return an immutable copy of current positions."""
        positions, _, _ = self._state()
        result = positions.copy()
        result.setflags(write=False)
        return result

    def reset(self, *, seed: int) -> ResetResult:
        """Create a deterministic noisy translated formation."""
        if self._closed:
            raise RuntimeError("cannot reset a closed environment.")
        rng = make_rng(seed, RandomStream.INITIAL_STATE)
        noise_limit = self._config.task.initial_position_noise_m
        noise = rng.uniform(
            -noise_limit,
            noise_limit,
            size=(len(self._agent_ids), 3),
        )
        initial_center = np.asarray(self._config.task.initial_center_m, dtype=np.float64)
        self._positions = self._relative_targets + initial_center + noise
        self._velocities = np.zeros_like(self._positions)
        self._previous_actions = np.zeros_like(self._positions)
        self._step_count = 0
        self._success_streak = 0
        self._episode_done = False
        return ResetResult(self._local_observations(), self._centralized_state())

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        """Advance first-order dynamics by one configured control interval."""
        positions, _, previous_actions = self._state()
        if self._episode_done:
            raise RuntimeError("episode is complete; call reset before step.")
        if tuple(actions.agent_ids) != self._agent_ids:
            raise ValueError("action agent_ids must exactly match environment row order.")

        normalized_actions = np.asarray(actions.values, dtype=np.float64)
        velocities = normalized_actions * self._config.environment.max_velocity_component_mps
        positions = positions + velocities * self._config.environment.time_step_seconds
        self._positions = positions
        self._velocities = velocities
        self._step_count += 1

        state_metrics = evaluate_kinematic_state(
            positions,
            self._targets,
            normalized_actions,
            previous_actions,
            collision_distance_m=self._config.task.collision_distance_m,
            success_tolerance_m=self._config.task.success_tolerance_m,
        )
        if state_metrics.within_tolerance:
            self._success_streak += 1
        else:
            self._success_streak = 0

        success = self._success_streak >= self._config.task.success_hold_steps
        collision_failure = (
            self._config.task.terminate_on_collision and state_metrics.collision_pairs > 0
        )
        terminated = success or collision_failure
        truncated = (
            not terminated and self._step_count >= self._config.environment.max_episode_steps
        )

        reward = compute_reward(
            positions,
            self._targets,
            normalized_actions,
            previous_actions,
            collision_distance_m=self._config.task.collision_distance_m,
            success=success,
            config=self._config.reward,
        )
        metrics = state_metrics.to_dict()
        metrics.update(reward.mean_metrics())
        metrics.update(
            {
                "success": float(success),
                "collision_failure": float(collision_failure),
                "step_count": float(self._step_count),
                "success_streak": float(self._success_streak),
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
        """Mark this lightweight environment closed."""
        self._closed = True

    def _state(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        if self._positions is None or self._velocities is None or self._previous_actions is None:
            raise RuntimeError("environment must be reset before use.")
        return self._positions, self._velocities, self._previous_actions

    def _local_observations(self) -> LocalObservations:
        positions, velocities, _ = self._state()
        relative_targets = self._targets - positions
        ego = np.concatenate((relative_targets, velocities), axis=1).astype(np.float32)

        num_agents = len(self._agent_ids)
        max_neighbors = self._config.observation.max_neighbors
        neighbors = np.zeros((num_agents, max_neighbors, 6), dtype=np.float32)
        mask = np.zeros((num_agents, max_neighbors), dtype=np.bool_)
        radius = self._config.observation.neighbor_radius_m

        for agent_index in range(num_agents):
            relative_positions = positions - positions[agent_index]
            distances = np.array(
                [
                    math.sqrt(sum(float(component) ** 2 for component in relative_position))
                    for relative_position in relative_positions
                ],
                dtype=np.float64,
            )
            candidates = [
                other_index
                for other_index in range(num_agents)
                if other_index != agent_index
                and (radius is None or distances[other_index] <= radius)
            ]
            candidates.sort(
                key=lambda index: (
                    round(float(distances[index]), 12),
                    self._agent_ids[index].value,
                )
            )
            for slot, neighbor_index in enumerate(candidates[:max_neighbors]):
                neighbors[agent_index, slot, :3] = relative_positions[neighbor_index]
                neighbors[agent_index, slot, 3:] = (
                    velocities[neighbor_index] - velocities[agent_index]
                )
                mask[agent_index, slot] = True

        return LocalObservations(self._agent_ids, ego, neighbors, mask)

    def _centralized_state(self) -> CentralizedState:
        positions, velocities, _ = self._state()
        values = np.concatenate(
            (positions.reshape(-1), velocities.reshape(-1), self._targets.reshape(-1))
        ).astype(np.float32)
        return CentralizedState(values)
