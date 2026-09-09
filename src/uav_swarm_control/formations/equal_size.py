"""Equal-cardinality templates for pooled multi-formation learning."""

import math

import numpy as np

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations._validation import agent_count, positive_scalar
from uav_swarm_control.formations.templates import FormationKind, create_formation


def _center_and_scale(points: FloatArray, spacing: float) -> FloatArray:
    centered = points - points.mean(axis=0, keepdims=True)
    differences = centered[:, None, :] - centered[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min())
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("template contains coincident points.")
    return centered * (spacing / minimum)


def rectangular_plane(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return a compact planar grid for any positive swarm size."""
    count = agent_count(num_agents)
    separation = positive_scalar(spacing, name="spacing")
    columns = math.ceil(math.sqrt(count))
    points = np.array(
        [(index % columns, index // columns, 0.0) for index in range(count)],
        dtype=np.float64,
    )
    if count == 1:
        return points
    return _center_and_scale(points, separation)


def generalized_pyramid(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return one apex above a compact base for any swarm of at least four UAVs."""
    count = agent_count(num_agents, minimum=4)
    separation = positive_scalar(spacing, name="spacing")
    base = rectangular_plane(count - 1, 1.0)
    base[:, 2] = -0.25
    apex = np.array([[0.0, 0.0, 0.75 + math.sqrt(count - 1) / 2.0]], dtype=np.float64)
    return _center_and_scale(np.concatenate((base, apex), axis=0), separation)


def vertical_column(num_agents: int, spacing: float = 1.0) -> FloatArray:
    """Return a narrow obstacle-passage shape with stable row correspondence."""
    count = agent_count(num_agents)
    separation = positive_scalar(spacing, name="spacing")
    z = np.arange(count, dtype=np.float64) * separation
    return np.column_stack((np.zeros(count), np.zeros(count), z - z.mean()))


def equal_size_formation(
    kind: FormationKind | str,
    num_agents: int,
    spacing: float = 1.0,
) -> FloatArray:
    """Create plane/cube/sphere/pyramid choices with one fixed actor row count."""
    formation = FormationKind(kind)
    if formation is FormationKind.PLANE:
        return rectangular_plane(num_agents, spacing)
    if formation is FormationKind.PYRAMID:
        return generalized_pyramid(num_agents, spacing)
    return create_formation(formation, num_agents, spacing)


__all__ = [
    "equal_size_formation",
    "generalized_pyramid",
    "rectangular_plane",
    "vertical_column",
]
