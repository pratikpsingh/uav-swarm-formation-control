"""Pooled-formation and construction/waypoint research environments."""

from collections.abc import Mapping

import numpy as np

from uav_swarm_control.communication import CommunicationRegimen
from uav_swarm_control.configuration.generalization import GeneralizationVariant
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.communication import CommunicationFormationEnvironment
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.environments.pybullet import BackendFactory
from uav_swarm_control.formations import FormationKind, FormationPoseRange
from uav_swarm_control.formations.equal_size import equal_size_formation
from uav_swarm_control.missions import (
    ConstructionMission,
    MissionPhase,
    WaypointTracker,
    linear_waypoints,
)
from uav_swarm_control.obstacles import ObstacleFieldConfig
from uav_swarm_control.seeding import RandomStream, make_rng


class PooledFormationEnvironment(CommunicationFormationEnvironment):
    """Sample one equal-cardinality 2D/3D formation per episode."""

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
        training_steps_per_environment: int | None = None,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        if not formation_kinds or len(set(formation_kinds)) != len(formation_kinds):
            raise ValueError("formation_kinds must be non-empty and unique.")
        self._formation_kinds = tuple(FormationKind(kind) for kind in formation_kinds)
        self._active_formation_kind = self._formation_kinds[0]
        super().__init__(
            config,
            pose_range,
            variant,
            obstacle_config,
            regimen,
            payload_bytes_per_neighbor=payload_bytes_per_neighbor,
            training_steps_per_environment=training_steps_per_environment,
            backend_factory=backend_factory,
        )

    @property
    def active_formation_kind(self) -> FormationKind:
        return self._active_formation_kind

    @property
    def episode_metadata(self) -> Mapping[str, float]:
        metadata = dict(super().episode_metadata)
        metadata["formation_index"] = float(
            self._formation_kinds.index(self._active_formation_kind)
        )
        return metadata

    def reset(self, *, seed: int) -> ResetResult:
        generator = make_rng(seed, RandomStream.FORMATION_POSE)
        self._active_formation_kind = self._formation_kinds[
            int(generator.integers(len(self._formation_kinds)))
        ]
        formation = self._config.experiment.formation
        self._template = equal_size_formation(
            self._active_formation_kind,
            formation.num_agents,
            formation.spacing_m,
        )
        return super().reset(seed=seed)


class MissionFormationEnvironment(PooledFormationEnvironment):
    """Construct near ground, hold formation, then follow three-dimensional waypoints."""

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
        initial_ground_altitude_m: float,
        construction_altitude_m: float,
        construction_hold_steps: int,
        waypoint_spacing_m: float,
        waypoint_hold_steps: int,
        training_steps_per_environment: int | None = None,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        if initial_ground_altitude_m <= 0.0 or construction_altitude_m <= initial_ground_altitude_m:
            raise ValueError("construction altitude must be above a positive ground altitude.")
        self._initial_ground_altitude_m = initial_ground_altitude_m
        self._construction_altitude_m = construction_altitude_m
        self._construction_hold_steps = construction_hold_steps
        self._waypoint_spacing_m = waypoint_spacing_m
        self._waypoint_hold_steps = waypoint_hold_steps
        self._mission = ConstructionMission(construction_hold_steps)
        self._waypoint_tracker: WaypointTracker | None = None
        self._formation_offsets = np.empty((0, 3), dtype=np.float64)
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
    def mission_phase(self) -> MissionPhase:
        return self._mission.phase

    @property
    def episode_metadata(self) -> Mapping[str, float]:
        metadata = dict(super().episode_metadata)
        metadata.update(
            {
                "waypoint_count": float(
                    0 if self._waypoint_tracker is None else len(self._waypoint_tracker.waypoints)
                ),
                "construction_altitude_m": self._construction_altitude_m,
                "initial_ground_altitude_m": self._initial_ground_altitude_m,
            }
        )
        return metadata

    def reset(self, *, seed: int) -> ResetResult:
        formation = self._config.experiment.formation
        ground = equal_size_formation(
            FormationKind.PLANE,
            formation.num_agents,
            formation.spacing_m,
        )
        initial_center = np.asarray(self._config.experiment.task.initial_center_m, dtype=np.float64)
        ground[:, 2] = self._initial_ground_altitude_m - initial_center[2]
        self._initial_relative_positions = ground
        super().reset(seed=seed)
        self._mission = ConstructionMission(self._construction_hold_steps)
        final_center = self._targets.mean(axis=0)
        self._formation_offsets = self._targets - final_center
        construction_center = np.array(
            [initial_center[0], initial_center[1], self._construction_altitude_m],
            dtype=np.float64,
        )
        self._waypoint_tracker = WaypointTracker(
            linear_waypoints(construction_center, final_center, self._waypoint_spacing_m),
            self._config.experiment.task.success_tolerance_m,
            self._waypoint_hold_steps,
        )
        self._targets = self._formation_offsets + construction_center
        self._success_streak = 0
        return ResetResult(self._local_observations(), self._centralized_state())

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        transition = super().step(actions)
        collision_failure = transition.metrics["collision_failure"] > 0.0
        tracker = self._require_tracker()
        formation_ready = (
            transition.metrics["position_rmse_m"]
            <= self._config.experiment.task.success_tolerance_m
        )
        previous_phase = self._mission.phase
        navigation_complete = False
        if previous_phase is MissionPhase.NAVIGATING:
            tracker.update(self.positions.mean(axis=0))
            self._targets = self._formation_offsets + tracker.current
            navigation_complete = tracker.complete and formation_ready
        phase = self._mission.update(
            formation_ready=formation_ready,
            navigation_complete=navigation_complete,
        )
        if previous_phase is MissionPhase.HOLDING and phase is MissionPhase.NAVIGATING:
            self._targets = self._formation_offsets + tracker.current
            self._success_streak = 0
        complete = phase is MissionPhase.COMPLETE
        terminated = collision_failure or complete
        truncated = transition.truncated and not terminated
        metrics = dict(transition.metrics)
        metrics.update(
            {
                "success": float(complete),
                "mission_phase_index": float(tuple(MissionPhase).index(phase)),
                "construction_success": float(
                    phase in {MissionPhase.NAVIGATING, MissionPhase.COMPLETE}
                ),
                "navigation_success": float(complete),
                "construction_steps": float(self._mission.construction_steps),
                "holding_steps": float(self._mission.holding_steps),
                "navigation_steps": float(self._mission.navigation_steps),
                "active_waypoint_index": float(tracker.index),
            }
        )
        self._episode_done = terminated or truncated
        return StepResult(
            self._local_observations(),
            self._centralized_state(),
            transition.rewards,
            terminated,
            truncated,
            metrics,
        )

    def _require_tracker(self) -> WaypointTracker:
        if self._waypoint_tracker is None:
            raise RuntimeError("environment must be reset before stepping.")
        return self._waypoint_tracker


__all__ = ["MissionFormationEnvironment", "PooledFormationEnvironment"]
