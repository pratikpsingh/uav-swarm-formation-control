"""Episode-randomized 3D formation task over the corrected physical reward."""

from collections.abc import Mapping

import numpy as np

from uav_swarm_control.configuration.generalization import (
    AssignmentMode,
    CoordinateFrame,
    GeneralizationVariant,
)
from uav_swarm_control.configuration.pybullet import PyBulletExperimentConfig
from uav_swarm_control.environments.contracts import (
    NormalizedVelocityActions,
    ResetResult,
    StepResult,
)
from uav_swarm_control.environments.gym_pybullet_drones import GymPyBulletDronesBackend
from uav_swarm_control.environments.paper04 import Paper04Environment
from uav_swarm_control.environments.pybullet import BackendFactory
from uav_swarm_control.formations import (
    FormationAssignment,
    FormationPose,
    FormationPoseRange,
    create_formation,
    fixed_assignment,
    formation_diameter,
    minimum_distance_assignment,
    sample_formation_pose,
    transform,
)
from uav_swarm_control.observations import (
    CentralizedState,
    LocalObservations,
    build_centralized_state,
)
from uav_swarm_control.seeding import RandomStream, make_rng


class GeneralizedFormationEnvironment(Paper04Environment):
    """Randomize target pose while keeping initial geometry independent and observable."""

    def __init__(
        self,
        config: PyBulletExperimentConfig,
        pose_range: FormationPoseRange,
        variant: GeneralizationVariant,
        *,
        backend_factory: BackendFactory = GymPyBulletDronesBackend,
    ) -> None:
        self._pose_range = pose_range
        self._variant = variant
        formation = config.experiment.formation
        self._template = create_formation(
            formation.kind,
            formation.num_agents,
            formation.spacing_m,
        )
        self._pose: FormationPose | None = None
        self._assignment: FormationAssignment | None = None
        super().__init__(config, backend_factory=backend_factory)

    @property
    def episode_metadata(self) -> Mapping[str, float]:
        """Return sampled pose and nominal assignment workload for the active episode."""
        if self._pose is None or self._assignment is None:
            raise RuntimeError("environment must be reset before reading episode metadata.")
        center = self._pose.center_m
        euler = self._pose.euler_radians
        count = len(self._assignment.target_indices)
        return {
            "target_center_x_m": center[0],
            "target_center_y_m": center[1],
            "target_center_z_m": center[2],
            "target_roll_rad": euler[0],
            "target_pitch_rad": euler[1],
            "target_yaw_rad": euler[2],
            "target_scale": self._pose.scale,
            "target_diameter_m": formation_diameter(self._targets),
            "nominal_assignment_total_distance_m": self._assignment.total_distance_m,
            "nominal_assignment_mean_distance_m": self._assignment.total_distance_m / count,
        }

    @property
    def target_indices(self) -> tuple[int, ...]:
        """Target-template row assigned to each stable agent row."""
        if self._assignment is None:
            raise RuntimeError("environment must be reset before reading assignment.")
        return self._assignment.target_indices

    def reset(self, *, seed: int) -> ResetResult:
        """Sample target pose/assignment independently from initial-state position noise."""
        formation = self._config.experiment.formation
        self._pose = sample_formation_pose(
            formation.center_m,
            formation.euler_radians,
            self._pose_range,
            make_rng(seed, RandomStream.FORMATION_POSE),
        )
        targets = transform(
            self._template,
            rotation=self._pose.rotation,
            offset=self._pose.center_m,
            scale_factor=self._pose.scale,
        )
        initial_center = np.asarray(self._config.experiment.task.initial_center_m)
        nominal_initial = self._initial_relative_positions + initial_center
        self._assignment = (
            fixed_assignment(nominal_initial, targets)
            if self._variant.assignment is AssignmentMode.FIXED
            else minimum_distance_assignment(nominal_initial, targets)
        )
        self._targets = self._assignment.assigned_targets
        return super().reset(seed=seed)

    def step(self, actions: NormalizedVelocityActions) -> StepResult:
        """Decode target-frame actions, then retain world-frame physical metrics."""
        pose = self._require_pose()
        unclipped = np.asarray(actions.values, dtype=np.float64)
        if self._variant.coordinate_frame is CoordinateFrame.TARGET:
            unclipped = unclipped @ pose.rotation.T
        clipped = np.clip(unclipped, -1.0, 1.0)
        clip_fraction = float(np.count_nonzero(clipped != unclipped) / clipped.size)
        transition = super().step(
            NormalizedVelocityActions(actions.agent_ids, clipped.astype(np.float32))
        )
        metrics = dict(transition.metrics)
        metrics["action_frame_clip_fraction"] = clip_fraction
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
        if self._variant.coordinate_frame is CoordinateFrame.WORLD:
            return observations
        rotation = self._require_pose().rotation
        ego = np.asarray(observations.ego, dtype=np.float64).copy()
        ego[:, :3] = ego[:, :3] @ rotation
        ego[:, 3:6] = ego[:, 3:6] @ rotation
        neighbors = np.asarray(observations.neighbors, dtype=np.float64).copy()
        neighbors[:, :, :3] = neighbors[:, :, :3] @ rotation
        neighbors[:, :, 3:6] = neighbors[:, :, 3:6] @ rotation
        return LocalObservations(
            observations.agent_ids,
            ego.astype(np.float32),
            neighbors.astype(np.float32),
            observations.neighbor_mask,
        )

    def _centralized_state(self) -> CentralizedState:
        if self._variant.coordinate_frame is CoordinateFrame.WORLD:
            return super()._centralized_state()
        state, _ = self._state()
        pose = self._require_pose()
        center = np.asarray(pose.center_m, dtype=np.float64)
        rotation = pose.rotation
        return build_centralized_state(
            (np.asarray(state.positions, dtype=np.float64) - center) @ rotation,
            np.asarray(state.linear_velocities_mps, dtype=np.float64) @ rotation,
            (self._targets - center) @ rotation,
        )

    def _require_pose(self) -> FormationPose:
        if self._pose is None:
            raise RuntimeError("environment must be reset before use.")
        return self._pose


__all__ = ["GeneralizedFormationEnvironment"]
