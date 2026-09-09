"""Obstacle-triggered active-target morphing environment."""

import numpy as np

from uav_swarm_control.communication import CommunicationRegimen
from uav_swarm_control.configuration.generalization import GeneralizationVariant
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.environments.pybullet import BackendFactory
from uav_swarm_control.environments.research import PooledFormationEnvironment
from uav_swarm_control.formations import (
    FormationKind,
    FormationPoseRange,
    minimum_distance_assignment,
    normalized_shape_rmse,
)
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations.equal_size import vertical_column
from uav_swarm_control.missions import (
    MorphingController,
    MorphingPhase,
    interpolate_targets,
)
from uav_swarm_control.obstacles import ObstacleFieldConfig


class MorphingFormationEnvironment(PooledFormationEnvironment):
    """Morph to a narrow column while an oracle obstacle blocks the travel corridor."""

    def __init__(
        self,
        config: PyBulletExperimentConfig,
        pose_range: FormationPoseRange,
        variant: GeneralizationVariant,
        obstacle_config: ObstacleFieldConfig,
        regimen: CommunicationRegimen,
        *,
        formation_kinds: tuple[FormationKind, ...],
        payload_bytes_per_neighbor: int,
        morph_rate_per_second: float,
        release_hold_steps: int,
        trigger_clearance_m: float,
        training_steps_per_environment: int | None = None,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        if trigger_clearance_m < 0.0:
            raise ValueError("trigger_clearance_m must be non-negative.")
        self._morph_rate = morph_rate_per_second
        self._release_hold = release_hold_steps
        self._trigger_clearance = trigger_clearance_m
        self._morph = MorphingController(morph_rate_per_second, release_hold_steps)
        self._nominal_targets: FloatArray
        self._avoidance_targets: FloatArray
        super().__init__(
            config,
            pose_range,
            variant,
            obstacle_config,
            regimen,
            formation_kinds=formation_kinds,
            payload_bytes_per_neighbor=payload_bytes_per_neighbor,
            training_steps_per_environment=training_steps_per_environment,
            backend_factory=backend_factory,
        )

    @property
    def morphing_phase(self) -> MorphingPhase:
        return self._morph.phase

    def reset(self, *, seed: int) -> ResetResult:
        reset = super().reset(seed=seed)
        self._morph = MorphingController(self._morph_rate, self._release_hold)
        self._nominal_targets = self._targets.copy()
        center = np.asarray(self._nominal_targets.mean(axis=0), dtype=np.float64)
        formation = self._config.experiment.formation
        avoidance = vertical_column(formation.num_agents, formation.spacing_m) + center
        self._avoidance_targets = minimum_distance_assignment(
            self._nominal_targets, avoidance
        ).assigned_targets
        return reset

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        blocked = self._corridor_blocked()
        self._morph.update(
            blocked=blocked,
            time_step_seconds=self._config.experiment.environment.time_step_seconds,
        )
        self._targets = interpolate_targets(
            self._nominal_targets, self._avoidance_targets, self._morph.alpha
        )
        transition = super().step(actions)
        metrics = dict(transition.metrics)
        metrics.update(
            {
                "morph_alpha": self._morph.alpha,
                "morph_phase_index": float(tuple(MorphingPhase).index(self._morph.phase)),
                "corridor_blocked": float(blocked),
                "deformation_steps": float(self._morph.deformation_steps),
                "avoidance_steps": float(self._morph.avoidance_steps),
                "restoration_steps": float(self._morph.restoration_steps),
                "original_normalized_shape_rmse": normalized_shape_rmse(
                    self.positions, self._nominal_targets
                ),
                "restored_nominal": float(
                    self._morph.phase is MorphingPhase.NOMINAL and self._morph.alpha == 0.0
                ),
            }
        )
        return StepResult(
            transition.observations,
            transition.centralized_state,
            transition.rewards,
            transition.terminated,
            transition.truncated,
            metrics,
        )

    def _corridor_blocked(self) -> bool:
        field = self.obstacle_field
        if field.count == 0:
            return False
        start = np.asarray(self.positions.mean(axis=0), dtype=np.float64)
        end = np.asarray(self._nominal_targets.mean(axis=0), dtype=np.float64)
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-12:
            distances = np.linalg.norm(field.positions_m - end, axis=1)
        else:
            offsets = field.positions_m - start
            fractions = np.clip(offsets @ segment / denominator, 0.0, 1.0)
            closest = start + fractions[:, None] * segment
            distances = np.linalg.norm(field.positions_m - closest, axis=1)
        return bool(np.any(distances <= field.radii_m + self._trigger_clearance))


__all__ = ["MorphingFormationEnvironment"]
