"""Tests for simulator-independent target-pose sampling."""

import math

import numpy as np
import pytest

from uav_swarm_control.formations import (
    FormationPoseRange,
    pose_ranges_overlap,
    sample_formation_pose,
)


def _pose_range(*, center_x: tuple[float, float]) -> FormationPoseRange:
    return FormationPoseRange(
        (center_x[0], -0.2, -0.1),
        (center_x[1], 0.2, 0.1),
        (-0.3, -0.4, -0.5),
        (0.3, 0.4, 0.5),
        0.8,
        1.2,
    )


def test_pose_sampling_is_seeded_bounded_and_proper() -> None:
    pose_range = _pose_range(center_x=(-0.2, 0.2))
    first = sample_formation_pose(
        (1.0, 2.0, 3.0),
        (0.1, 0.2, 0.3),
        pose_range,
        np.random.default_rng(17),
    )
    second = sample_formation_pose(
        (1.0, 2.0, 3.0),
        (0.1, 0.2, 0.3),
        pose_range,
        np.random.default_rng(17),
    )

    assert first.center_m == second.center_m
    assert first.euler_radians == second.euler_radians
    assert first.scale == second.scale
    np.testing.assert_array_equal(first.rotation, second.rotation)
    assert 0.8 <= first.scale <= 1.2
    assert 0.8 <= first.center_m[0] <= 1.2
    np.testing.assert_allclose(first.rotation.T @ first.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(first.rotation) == pytest.approx(1.0)


def test_pose_range_overlap_uses_all_seven_dimensions() -> None:
    training = _pose_range(center_x=(-0.2, 0.2))
    touching = _pose_range(center_x=(0.2, 0.4))
    separate = _pose_range(center_x=(0.21, 0.4))

    assert pose_ranges_overlap(training, touching)
    assert not pose_ranges_overlap(training, separate)


@pytest.mark.parametrize(
    "pose_range",
    [
        FormationPoseRange((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), 1, 1),
    ],
)
def test_exact_pose_range_is_supported(pose_range: FormationPoseRange) -> None:
    pose = sample_formation_pose(
        (0.0, 0.0, 2.0),
        (0.0, 0.0, math.pi / 2),
        pose_range,
        np.random.default_rng(1),
    )
    assert pose.center_m == (0.0, 0.0, 2.0)
    assert pose.euler_radians == (0.0, 0.0, math.pi / 2)
    assert pose.scale == 1.0


def test_pose_range_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="0 < lower"):
        FormationPoseRange((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, 1)
