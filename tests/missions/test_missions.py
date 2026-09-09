"""Tests for waypoint, construction, and morphing mission logic."""

import numpy as np

from uav_swarm_control.missions import (
    ConstructionMission,
    MissionPhase,
    MorphingController,
    MorphingPhase,
    WaypointTracker,
    interpolate_targets,
    linear_waypoints,
)


def test_waypoints_include_goal_and_advance_after_hold() -> None:
    points = linear_waypoints([0.0, 0.0, 0.0], [2.5, 0.0, 0.0], 1.0)
    np.testing.assert_array_equal(points[-1], [2.5, 0.0, 0.0])
    tracker = WaypointTracker(points, tolerance_m=0.1, hold_steps=2)
    assert not tracker.update(points[0])
    assert tracker.update(points[0])
    assert tracker.index == 1


def test_construction_mission_separates_phases() -> None:
    mission = ConstructionMission(2)
    assert mission.update(formation_ready=True) is MissionPhase.HOLDING
    assert mission.update(formation_ready=True) is MissionPhase.HOLDING
    assert mission.update(formation_ready=True) is MissionPhase.NAVIGATING
    assert mission.update(formation_ready=True, navigation_complete=True) is MissionPhase.COMPLETE


def test_morphing_deforms_then_restores_with_bounded_alpha() -> None:
    controller = MorphingController(alpha_rate_per_second=1.0, release_hold_steps=1)
    assert controller.update(blocked=True, time_step_seconds=0.5) is MorphingPhase.DEFORMING
    assert controller.alpha == 0.5
    assert controller.update(blocked=True, time_step_seconds=0.5) is MorphingPhase.AVOIDING
    assert controller.update(blocked=False, time_step_seconds=0.5) is MorphingPhase.RESTORING
    controller.update(blocked=False, time_step_seconds=0.5)
    assert controller.alpha == 0.5
    assert controller.update(blocked=False, time_step_seconds=0.5) is MorphingPhase.NOMINAL
    np.testing.assert_array_equal(
        interpolate_targets(np.zeros((2, 3)), np.ones((2, 3)), 0.25),
        np.full((2, 3), 0.25),
    )
