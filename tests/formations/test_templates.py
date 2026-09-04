"""Behavioral tests for formation templates."""

import numpy as np
import pytest
from numpy.typing import NDArray

from uav_swarm_control.formations import FormationKind, create_formation


def minimum_separation(points: NDArray[np.float64]) -> float:
    """Return the smallest distance between different points."""
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


@pytest.mark.parametrize(
    ("kind", "num_agents"),
    [
        (FormationKind.LINE, 4),
        (FormationKind.TRIANGLE, 3),
        (FormationKind.SQUARE, 4),
        (FormationKind.POLYGON, 7),
        (FormationKind.PLANE, 9),
        (FormationKind.CUBE, 8),
        (FormationKind.SPHERE, 10),
        (FormationKind.PYRAMID, 5),
    ],
)
def test_templates_are_centered_float64_arrays(
    kind: FormationKind,
    num_agents: int,
) -> None:
    """Every public template follows the shared representation contract."""
    points = create_formation(kind, num_agents, spacing=0.6)

    assert points.shape == (num_agents, 3)
    assert points.dtype == np.float64
    np.testing.assert_allclose(points.mean(axis=0), np.zeros(3), atol=1e-12)


@pytest.mark.parametrize(
    ("kind", "num_agents"),
    [
        (FormationKind.LINE, 4),
        (FormationKind.TRIANGLE, 3),
        (FormationKind.SQUARE, 4),
        (FormationKind.POLYGON, 7),
        (FormationKind.PLANE, 9),
        (FormationKind.CUBE, 8),
        (FormationKind.SPHERE, 10),
        (FormationKind.PYRAMID, 5),
    ],
)
def test_spacing_always_means_minimum_separation(
    kind: FormationKind,
    num_agents: int,
) -> None:
    """Changing formation kind must not change the unit of spacing."""
    spacing = 0.6
    points = create_formation(kind, num_agents, spacing)

    assert minimum_separation(points) == pytest.approx(spacing)


def test_single_agent_line_is_at_origin() -> None:
    """A singleton has a useful location even though separation is undefined."""
    points = create_formation(FormationKind.LINE, 1, spacing=0.6)

    np.testing.assert_array_equal(points, np.zeros((1, 3)))


@pytest.mark.parametrize(
    ("kind", "num_agents", "message"),
    [
        (FormationKind.TRIANGLE, 4, "must be 3"),
        (FormationKind.SQUARE, 3, "must be 4"),
        (FormationKind.PLANE, 5, "perfect-square"),
        (FormationKind.CUBE, 9, "perfect-cube"),
        (FormationKind.PYRAMID, 6, "must be 5"),
    ],
)
def test_ambiguous_agent_counts_are_rejected(
    kind: FormationKind,
    num_agents: int,
    message: str,
) -> None:
    """Named geometric structures should not silently become partial shapes."""
    with pytest.raises(ValueError, match=message):
        create_formation(kind, num_agents)


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.inf, np.nan])
def test_invalid_spacing_is_rejected(spacing: float) -> None:
    """Spacing must represent a positive finite physical distance."""
    with pytest.raises(ValueError, match="spacing"):
        create_formation(FormationKind.TRIANGLE, 3, spacing)


def test_unknown_formation_kind_is_rejected() -> None:
    """Configuration mistakes should fail before an experiment starts."""
    with pytest.raises(ValueError, match="unknown formation kind"):
        create_formation("hexagon", 6)
