"""Deterministic, centered formation templates with consistent spacing."""

import math
from enum import StrEnum

import numpy as np

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations._validation import agent_count, positive_scalar


class FormationKind(StrEnum):
    """Names accepted by create_formation."""

    LINE = "line"
    TRIANGLE = "triangle"
    SQUARE = "square"
    POLYGON = "polygon"
    PLANE = "plane"
    CUBE = "cube"
    SPHERE = "sphere"
    PYRAMID = "pyramid"


def _center(points: FloatArray) -> FloatArray:
    """Center a newly created template at the origin."""
    return points - points.mean(axis=0, keepdims=True)


def _scale_minimum_distance(points: FloatArray, spacing: float) -> FloatArray:
    """Scale a non-degenerate template to the requested minimum separation."""
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    minimum_distance = float(distances.min())
    if not math.isfinite(minimum_distance) or minimum_distance <= 0.0:
        raise ValueError("formation contains coincident points and cannot be scaled.")
    return points * (spacing / minimum_distance)


def line(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return equally spaced points on the x-axis."""
    count = agent_count(num_agents)
    separation = positive_scalar(spacing, name="spacing")
    x_coordinates = np.arange(count, dtype=np.float64) * separation
    points = np.column_stack((x_coordinates, np.zeros((count, 2), dtype=np.float64)))
    return _center(points)


def regular_polygon(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return a regular polygon in the xy-plane with the requested edge spacing."""
    count = agent_count(num_agents, minimum=3)
    separation = positive_scalar(spacing, name="spacing")
    radius = separation / (2.0 * math.sin(math.pi / count))
    angles = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
    points = np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(count, dtype=np.float64),
        )
    )
    return _center(points)


def triangle(spacing: float = 1.0) -> FloatArray:
    """Return an equilateral triangle."""
    return regular_polygon(3, spacing)


def square(spacing: float = 1.0) -> FloatArray:
    """Return a square."""
    return regular_polygon(4, spacing)


def square_grid(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return a complete square lattice in the xy-plane."""
    count = agent_count(num_agents)
    separation = positive_scalar(spacing, name="spacing")
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError(
            "plane formation requires a perfect-square num_agents "
            f"(1, 4, 9, ...); received {count}."
        )
    coordinates = np.arange(side, dtype=np.float64) * separation
    points = np.array(
        [(x, y, 0.0) for x in coordinates for y in coordinates],
        dtype=np.float64,
    )
    return _center(points)


def cubic_lattice(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return a complete cubic lattice."""
    count = agent_count(num_agents)
    separation = positive_scalar(spacing, name="spacing")
    side = 1
    while side**3 < count:
        side += 1
    if side**3 != count:
        raise ValueError(
            f"cube formation requires a perfect-cube num_agents (1, 8, 27, ...); received {count}."
        )
    coordinates = np.arange(side, dtype=np.float64) * separation
    points = np.array(
        [(x, y, z) for x in coordinates for y in coordinates for z in coordinates],
        dtype=np.float64,
    )
    return _center(points)


def spherical_shell(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return approximately uniform points around a spherical shell.

    Golden-angle sampling supports arbitrary counts. The finite point set is
    centered and scaled so its minimum pairwise distance equals spacing.
    """
    count = agent_count(num_agents, minimum=2)
    separation = positive_scalar(spacing, name="spacing")
    indices = np.arange(count, dtype=np.float64)
    y_coordinates = 1.0 - 2.0 * indices / (count - 1)
    radial_coordinates = np.sqrt(np.maximum(0.0, 1.0 - y_coordinates**2))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angles = golden_angle * indices
    points = np.column_stack(
        (
            radial_coordinates * np.cos(angles),
            y_coordinates,
            radial_coordinates * np.sin(angles),
        )
    )
    return _scale_minimum_distance(_center(points), separation)


def square_pyramid(num_agents: int = 5, spacing: float = 1.0) -> FloatArray:
    """Return a regular square pyramid with four base vertices and one apex."""
    agent_count(num_agents, exact=5)
    separation = positive_scalar(spacing, name="spacing")
    half_side = separation / 2.0
    height = separation / math.sqrt(2.0)
    points = np.array(
        [
            [-half_side, -half_side, 0.0],
            [half_side, -half_side, 0.0],
            [half_side, half_side, 0.0],
            [-half_side, half_side, 0.0],
            [0.0, 0.0, height],
        ],
        dtype=np.float64,
    )
    return _center(points)


def create_formation(
    kind: FormationKind | str,
    num_agents: int,
    spacing: float = 1.0,
) -> FloatArray:
    """Create a validated formation template by name."""
    try:
        formation_kind = FormationKind(kind)
    except ValueError as error:
        choices = ", ".join(item.value for item in FormationKind)
        raise ValueError(f"unknown formation kind {kind!r}; choose one of: {choices}.") from error

    if formation_kind is FormationKind.LINE:
        return line(num_agents, spacing)
    if formation_kind is FormationKind.TRIANGLE:
        agent_count(num_agents, exact=3)
        return triangle(spacing)
    if formation_kind is FormationKind.SQUARE:
        agent_count(num_agents, exact=4)
        return square(spacing)
    if formation_kind is FormationKind.POLYGON:
        return regular_polygon(num_agents, spacing)
    if formation_kind is FormationKind.PLANE:
        return square_grid(num_agents, spacing)
    if formation_kind is FormationKind.CUBE:
        return cubic_lattice(num_agents, spacing)
    if formation_kind is FormationKind.SPHERE:
        return spherical_shell(num_agents, spacing)
    return square_pyramid(num_agents, spacing)


__all__ = [
    "FormationKind",
    "create_formation",
    "cubic_lattice",
    "line",
    "regular_polygon",
    "spherical_shell",
    "square",
    "square_grid",
    "square_pyramid",
    "triangle",
]
