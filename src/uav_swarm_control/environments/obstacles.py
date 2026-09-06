"""Oracle spherical-obstacle extension of the generalized formation task."""

from collections.abc import Mapping

import numpy as np

from uav_swarm_control.configuration.generalization import CoordinateFrame, GeneralizationVariant
from uav_swarm_control.configuration.obstacles import CurriculumPhase, scenario_for_progress
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.generalization import GeneralizedFormationEnvironment
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.environments.pybullet import BackendFactory
from uav_swarm_control.formations import FormationPoseRange
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.observations import CentralizedState, LocalObservations
from uav_swarm_control.obstacles import (
    ObstacleField,
    ObstacleFieldConfig,
    ObstacleScenario,
    advance_obstacles,
    empty_obstacle_field,
    obstacle_safety,
    sample_obstacle_field,
)
from uav_swarm_control.seeding import RandomStream, make_rng


class ObstacleFormationEnvironment(GeneralizedFormationEnvironment):
    """Expose exact obstacle state and penalize sampled sphere intersections."""

    def __init__(
        self,
        config: PyBulletExperimentConfig,
        pose_range: FormationPoseRange,
        variant: GeneralizationVariant,
        obstacle_config: ObstacleFieldConfig,
        phases: tuple[CurriculumPhase, ...],
        *,
        training_steps_per_environment: int | None = None,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        if not phases:
            raise ValueError("at least one obstacle phase is required.")
        if training_steps_per_environment is not None and training_steps_per_environment < 1:
            raise ValueError("training_steps_per_environment must be positive.")
        if training_steps_per_environment is None and len(phases) != 1:
            raise ValueError("evaluation environments require exactly one fixed scenario.")
        self._obstacle_config = obstacle_config
        self._phases = phases
        self._training_steps = training_steps_per_environment
        self._elapsed_training_steps = 0
        self._scenario = phases[0].scenario
        self._obstacle_field = empty_obstacle_field()
        self._episode_obstacle_collision = False
        self._episode_minimum_clearance_m: float | None = None
        super().__init__(config, pose_range, variant, backend_factory=backend_factory)

    @property
    def obstacle_field(self) -> ObstacleField:
        """Return the immutable active oracle field."""
        return self._obstacle_field

    @property
    def obstacle_scenario(self) -> ObstacleScenario:
        """Scenario selected at the most recent reset."""
        return self._scenario

    @property
    def episode_metadata(self) -> Mapping[str, float]:
        metadata = dict(super().episode_metadata)
        metadata.update(
            {
                "obstacle_scenario_index": float(tuple(ObstacleScenario).index(self._scenario)),
                "obstacle_count": float(self._obstacle_field.count),
                "dynamic_obstacle_count": float(self._obstacle_field.dynamic_count),
                "curriculum_progress": self._curriculum_progress(),
            }
        )
        return metadata

    def reset(self, *, seed: int) -> ResetResult:
        self._scenario = scenario_for_progress(self._phases, self._curriculum_progress())
        self._obstacle_field = empty_obstacle_field()
        super().reset(seed=seed)
        self._obstacle_field = sample_obstacle_field(
            self._scenario,
            self.positions,
            self.target_positions,
            self._obstacle_config,
            make_rng(seed, RandomStream.OBSTACLES),
        )
        safety = obstacle_safety(
            self.positions,
            self._obstacle_field,
            vehicle_radius_m=self._obstacle_config.vehicle_radius_m,
        )
        self._episode_obstacle_collision = safety.collision_pairs > 0
        self._episode_minimum_clearance_m = safety.minimum_clearance_m
        return ResetResult(self._local_observations(), self._centralized_state())

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        self._obstacle_field = advance_obstacles(
            self._obstacle_field, self._config.experiment.environment.time_step_seconds
        )
        transition = super().step(actions)
        self._elapsed_training_steps += 1
        safety = obstacle_safety(
            self.positions,
            self._obstacle_field,
            vehicle_radius_m=self._obstacle_config.vehicle_radius_m,
        )
        if safety.minimum_clearance_m is None:
            violation = np.zeros(len(self.agent_ids), dtype=np.float64)
        else:
            nearest = np.min(safety.clearances_m, axis=1)
            violation = np.clip(
                (self._obstacle_config.safety_margin_m - nearest)
                / self._obstacle_config.safety_margin_m,
                0.0,
                1.0,
            )
        proximity = self._obstacle_config.proximity_weight * np.square(violation)
        collision = self._obstacle_config.collision_penalty * safety.collided_agents.astype(
            np.float64
        )
        rewards = np.asarray(transition.rewards, dtype=np.float64) - proximity - collision
        obstacle_collision = safety.collision_pairs > 0
        self._episode_obstacle_collision |= obstacle_collision
        if safety.minimum_clearance_m is not None:
            self._episode_minimum_clearance_m = (
                safety.minimum_clearance_m
                if self._episode_minimum_clearance_m is None
                else min(self._episode_minimum_clearance_m, safety.minimum_clearance_m)
            )
        metrics = dict(transition.metrics)
        metrics.update(
            {
                "reward/obstacle_proximity_mean": -float(proximity.mean()),
                "reward/obstacle_collision_mean": -float(collision.mean()),
                "reward/total_mean": float(rewards.mean()),
                "obstacle_count": float(self._obstacle_field.count),
                "dynamic_obstacle_count": float(self._obstacle_field.dynamic_count),
                "obstacle_collision_pairs": float(safety.collision_pairs),
                "obstacle_collision_failure": float(obstacle_collision),
                "obstacle_collision_episode": float(self._episode_obstacle_collision),
            }
        )
        if safety.minimum_clearance_m is not None:
            metrics["minimum_obstacle_clearance_m"] = safety.minimum_clearance_m
        if self._episode_minimum_clearance_m is not None:
            metrics["episode_minimum_obstacle_clearance_m"] = self._episode_minimum_clearance_m
        terminated = transition.terminated or obstacle_collision
        truncated = transition.truncated and not obstacle_collision
        if obstacle_collision:
            metrics["success"] = 0.0
            self._episode_done = True
        return StepResult(
            transition.observations,
            transition.centralized_state,
            rewards.astype(np.float32),
            terminated,
            truncated,
            metrics,
        )

    def _local_observations(self) -> LocalObservations:
        observations = super()._local_observations()
        state, _ = self._state()
        positions: FloatArray = np.asarray(state.positions, dtype=np.float64)
        velocities: FloatArray = np.asarray(state.linear_velocities_mps, dtype=np.float64)
        maximum = self._obstacle_config.max_obstacles
        features = np.zeros((len(self.agent_ids), maximum, 7), dtype=np.float32)
        mask = np.zeros((len(self.agent_ids), maximum), dtype=np.bool_)
        for agent in range(len(self.agent_ids)):
            if self._obstacle_field.count == 0:
                continue
            relative: FloatArray = self._obstacle_field.positions_m - positions[agent]
            distances: FloatArray = np.linalg.norm(relative, axis=1)
            indices = np.argsort(distances, kind="stable")
            observation_radius = self._obstacle_config.observation_radius_m
            if observation_radius is not None:
                indices = indices[distances[indices] <= observation_radius]
            indices = indices[:maximum]
            slot_count = len(indices)
            relative_velocity = self._obstacle_field.velocities_mps[indices] - velocities[agent]
            relative_position = relative[indices]
            if self._variant.coordinate_frame is CoordinateFrame.TARGET:
                rotation = self._require_pose().rotation
                relative_position = relative_position @ rotation
                relative_velocity = relative_velocity @ rotation
            features[agent, :slot_count, :3] = relative_position
            features[agent, :slot_count, 3:6] = relative_velocity
            features[agent, :slot_count, 6] = self._obstacle_field.radii_m[indices]
            mask[agent, :slot_count] = True
        return LocalObservations(
            observations.agent_ids,
            observations.ego,
            observations.neighbors,
            observations.neighbor_mask,
            features,
            mask,
        )

    def _centralized_state(self) -> CentralizedState:
        base = super()._centralized_state()
        maximum = self._obstacle_config.max_obstacles
        positions = np.zeros((maximum, 3), dtype=np.float64)
        velocities = np.zeros((maximum, 3), dtype=np.float64)
        radii = np.zeros(maximum, dtype=np.float64)
        mask = np.zeros(maximum, dtype=np.float64)
        active = self._obstacle_field.count
        if active:
            positions[:active] = self._obstacle_field.positions_m
            velocities[:active] = self._obstacle_field.velocities_mps
            if self._variant.coordinate_frame is CoordinateFrame.TARGET:
                pose = self._require_pose()
                center = np.asarray(pose.center_m, dtype=np.float64)
                positions[:active] = (positions[:active] - center) @ pose.rotation
                velocities[:active] = velocities[:active] @ pose.rotation
            radii[:active] = self._obstacle_field.radii_m
            mask[:active] = 1.0
        values = np.concatenate((base.values, positions.ravel(), velocities.ravel(), radii, mask))
        return CentralizedState(values.astype(np.float32))

    def _curriculum_progress(self) -> float:
        if self._training_steps is None:
            return 0.0
        return min(self._elapsed_training_steps / self._training_steps, 1.0)


__all__ = ["ObstacleFormationEnvironment"]
