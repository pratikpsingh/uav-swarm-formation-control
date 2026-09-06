"""Dynamic communication-topology extension of the obstacle formation task."""

from collections.abc import Mapping

import numpy as np

from uav_swarm_control.communication import (
    CommunicationCondition,
    CommunicationRegimen,
    communication_graph_metrics,
)
from uav_swarm_control.configuration.generalization import GeneralizationVariant
from uav_swarm_control.configuration.obstacles import CurriculumPhase
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.environments.obstacles import ObstacleFormationEnvironment
from uav_swarm_control.environments.pybullet import BackendFactory
from uav_swarm_control.formations import FormationPoseRange
from uav_swarm_control.observations import LocalObservations
from uav_swarm_control.obstacles import ObstacleFieldConfig
from uav_swarm_control.seeding import RandomStream, make_rng


class CommunicationFormationEnvironment(ObstacleFormationEnvironment):
    """Apply one sampled communication condition while preserving padded shapes."""

    def __init__(
        self,
        config: PyBulletExperimentConfig,
        pose_range: FormationPoseRange,
        variant: GeneralizationVariant,
        obstacle_config: ObstacleFieldConfig,
        regimen: CommunicationRegimen,
        *,
        payload_bytes_per_neighbor: int,
        training_steps_per_environment: int | None = None,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        capacity = config.experiment.observation.max_neighbors
        if any(condition.requested_neighbors > capacity for condition in regimen.conditions):
            raise ValueError("communication condition exceeds observation capacity.")
        if payload_bytes_per_neighbor < 0:
            raise ValueError("payload_bytes_per_neighbor must be non-negative.")
        self._communication_regimen = regimen
        self._active_condition = regimen.conditions[0]
        self._payload_bytes_per_neighbor = payload_bytes_per_neighbor
        super().__init__(
            config,
            pose_range,
            variant,
            obstacle_config,
            (CurriculumPhase(1.0, self._active_condition.obstacle_scenario),),
            training_steps_per_environment=training_steps_per_environment,
            backend_factory=backend_factory,
        )

    @property
    def communication_condition(self) -> CommunicationCondition:
        """Condition selected for the current episode."""
        return self._active_condition

    @property
    def episode_metadata(self) -> Mapping[str, float]:
        metadata = dict(super().episode_metadata)
        radius = self._active_condition.sensing_radius_m
        metadata.update(
            {
                "requested_neighbors": float(self._active_condition.requested_neighbors),
                "sensing_radius_m": -1.0 if radius is None else radius,
                "unlimited_sensing": float(radius is None),
                "payload_bytes_per_neighbor": float(self._payload_bytes_per_neighbor),
            }
        )
        return metadata

    def reset(self, *, seed: int) -> ResetResult:
        generator = make_rng(seed, RandomStream.COMMUNICATION)
        index = int(generator.integers(len(self._communication_regimen.conditions)))
        self._active_condition = self._communication_regimen.conditions[index]
        self._phases = (CurriculumPhase(1.0, self._active_condition.obstacle_scenario),)
        return super().reset(seed=seed)

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        transition = super().step(actions)
        graph = communication_graph_metrics(
            self.positions,
            requested_neighbors=self._active_condition.requested_neighbors,
            sensing_radius_m=self._active_condition.sensing_radius_m,
            payload_bytes_per_neighbor=self._payload_bytes_per_neighbor,
        )
        metrics = dict(transition.metrics)
        metrics.update(graph.as_metrics())
        return StepResult(
            transition.observations,
            transition.centralized_state,
            transition.rewards,
            transition.terminated,
            transition.truncated,
            metrics,
        )

    def _local_observations(self) -> LocalObservations:
        observations = super()._local_observations()
        neighbors = np.array(observations.neighbors, copy=True)
        mask = np.zeros_like(observations.neighbor_mask)
        limit = self._active_condition.requested_neighbors
        radius = self._active_condition.sensing_radius_m
        for agent in range(len(self.agent_ids)):
            valid = np.flatnonzero(observations.neighbor_mask[agent])
            if radius is not None:
                distances = np.linalg.norm(neighbors[agent, valid, :3], axis=1)
                valid = valid[distances <= radius]
            valid = valid[:limit]
            mask[agent, valid] = True
            neighbors[agent, ~mask[agent]] = 0.0
        return LocalObservations(
            observations.agent_ids,
            observations.ego,
            neighbors,
            mask,
            observations.obstacles,
            observations.obstacle_mask,
        )


__all__ = ["CommunicationFormationEnvironment"]
