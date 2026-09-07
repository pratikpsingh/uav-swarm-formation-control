"""Corrected paper-style reward on the common velocity-controlled UAV task."""

import numpy as np

from uav_swarm_control.configuration import PyBulletExperimentConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    StepResult,
)
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.environments.pybullet import BackendFactory, PyBulletSwarmEnvironment
from uav_swarm_control.evaluation.kinematics import collision_statistics


class FormationProgressEnvironment(PyBulletSwarmEnvironment):
    """Use progress reward (Eq. 24) with corrected per-agent formation normalization."""

    def __init__(
        self,
        config: PyBulletExperimentConfig,
        *,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        if config.experiment.reward.smoothness_weight or config.experiment.reward.success_bonus:
            raise ValueError("paper reward excludes smoothness shaping and success bonus.")
        super().__init__(config, backend_factory=backend_factory)
        self._navigation_weight = config.experiment.reward.navigation_weight
        self._collision_penalty = config.experiment.reward.collision_penalty
        self._collision_distance = config.experiment.task.collision_distance_m

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        previous_distances = np.linalg.norm(self.positions - self.target_positions, axis=1)
        transition = super().step(actions)
        distances = np.linalg.norm(self.positions - self.target_positions, axis=1)
        progress = previous_distances - distances
        collisions = collision_statistics(
            self.positions, collision_distance_m=self._collision_distance
        )
        rewards = (
            self._navigation_weight * progress
            + transition.metrics["reward/formation_mean"]
            - self._collision_penalty * collisions.collided_agents
        )
        metrics = dict(transition.metrics)
        metrics["reward/navigation_mean"] = float(self._navigation_weight * progress.mean())
        metrics["reward/total_mean"] = float(rewards.mean())
        return StepResult(
            transition.observations,
            transition.centralized_state,
            rewards.astype(np.float32),
            transition.terminated,
            transition.truncated,
            metrics,
        )
