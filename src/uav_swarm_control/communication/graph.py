"""Pure communication-graph, connectivity, and rigidity measurements."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_swarm_control.formations._validation import points_array

BoolMatrix = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CommunicationGraphMetrics:
    """Step-level descriptors of the directed sensing graph."""

    actual_degree_mean: float
    actual_degree_min: float
    actual_degree_max: float
    directed_edge_count: int
    reciprocal_edge_fraction: float
    connected_components: int
    connected: bool
    algebraic_connectivity: float
    intrinsic_dimension: int
    rigidity_rank: int
    maximum_rigidity_rank: int
    rigidity_rank_fraction: float
    infinitesimally_rigid: bool
    received_payload_bytes: int

    def as_metrics(self) -> dict[str, float]:
        """Return finite values suitable for environment metric records."""
        return {
            "communication/actual_degree_mean": self.actual_degree_mean,
            "communication/actual_degree_min": self.actual_degree_min,
            "communication/actual_degree_max": self.actual_degree_max,
            "communication/directed_edge_count": float(self.directed_edge_count),
            "communication/reciprocal_edge_fraction": self.reciprocal_edge_fraction,
            "communication/connected_components": float(self.connected_components),
            "communication/connected": float(self.connected),
            "communication/algebraic_connectivity": self.algebraic_connectivity,
            "communication/intrinsic_dimension": float(self.intrinsic_dimension),
            "communication/rigidity_rank": float(self.rigidity_rank),
            "communication/maximum_rigidity_rank": float(self.maximum_rigidity_rank),
            "communication/rigidity_rank_fraction": self.rigidity_rank_fraction,
            "communication/infinitesimally_rigid": float(self.infinitesimally_rigid),
            "communication/received_payload_bytes": float(self.received_payload_bytes),
        }


def build_neighbor_adjacency(
    positions: ArrayLike,
    *,
    requested_neighbors: int,
    sensing_radius_m: float | None,
) -> BoolMatrix:
    """Build the directed nearest-neighbor sensing graph with stable index ties."""
    points = points_array(positions, name="positions")
    count = points.shape[0]
    if isinstance(requested_neighbors, bool) or not 0 <= requested_neighbors <= count - 1:
        raise ValueError("requested_neighbors must lie in [0, num_agents - 1].")
    if sensing_radius_m is not None and (
        not math.isfinite(sensing_radius_m) or sensing_radius_m <= 0.0
    ):
        raise ValueError("sensing_radius_m must be positive when provided.")
    adjacency = np.zeros((count, count), dtype=np.bool_)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for agent in range(count):
        candidates = [
            other
            for other in range(count)
            if other != agent
            and (sensing_radius_m is None or distances[agent, other] <= sensing_radius_m)
        ]
        candidates.sort(key=lambda other: (round(float(distances[agent, other]), 12), other))
        adjacency[agent, candidates[:requested_neighbors]] = True
    return adjacency


def _components(adjacency: BoolMatrix) -> int:
    remaining = set(range(adjacency.shape[0]))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbors = {int(item) for item in np.flatnonzero(adjacency[current])} & remaining
            remaining -= neighbors
            stack.extend(neighbors)
    return count


def _rigidity(points: NDArray[np.float64], adjacency: BoolMatrix) -> tuple[int, int, int]:
    centered = points - points.mean(axis=0)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    tolerance = (
        max(centered.shape)
        * np.finfo(np.float64).eps
        * max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    )
    dimension = int(np.count_nonzero(singular_values > tolerance))
    if dimension == 0:
        return 0, 0, 0
    coordinates = centered @ right[:dimension].T
    edges = np.transpose(np.triu(adjacency, k=1).nonzero())
    matrix = np.zeros((len(edges), points.shape[0] * dimension), dtype=np.float64)
    for row, (left, right_index) in enumerate(edges):
        difference = coordinates[left] - coordinates[right_index]
        matrix[row, left * dimension : (left + 1) * dimension] = difference
        matrix[row, right_index * dimension : (right_index + 1) * dimension] = -difference
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    maximum = max(
        0,
        dimension * points.shape[0] - dimension * (dimension + 1) // 2,
    )
    return dimension, rank, maximum


def communication_graph_metrics(
    positions: ArrayLike,
    *,
    requested_neighbors: int,
    sensing_radius_m: float | None,
    payload_bytes_per_neighbor: int,
) -> CommunicationGraphMetrics:
    """Measure graph structure and observation-payload cost at one control step."""
    points = points_array(positions, name="positions")
    if isinstance(payload_bytes_per_neighbor, bool) or payload_bytes_per_neighbor < 0:
        raise ValueError("payload_bytes_per_neighbor must be a non-negative integer.")
    directed = build_neighbor_adjacency(
        points,
        requested_neighbors=requested_neighbors,
        sensing_radius_m=sensing_radius_m,
    )
    undirected = directed | directed.T
    degrees = directed.sum(axis=1)
    directed_edges = int(degrees.sum())
    reciprocal = int(np.count_nonzero(directed & directed.T))
    components = _components(undirected)
    laplacian = np.diag(undirected.sum(axis=1)) - undirected.astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    algebraic = max(0.0, float(eigenvalues[1])) if len(eigenvalues) > 1 else 0.0
    dimension, rigidity_rank, maximum_rank = _rigidity(points, undirected)
    rigidity_fraction = rigidity_rank / maximum_rank if maximum_rank else 0.0
    return CommunicationGraphMetrics(
        actual_degree_mean=float(degrees.mean()),
        actual_degree_min=float(degrees.min()),
        actual_degree_max=float(degrees.max()),
        directed_edge_count=directed_edges,
        reciprocal_edge_fraction=(reciprocal / directed_edges if directed_edges else 0.0),
        connected_components=components,
        connected=components == 1,
        algebraic_connectivity=algebraic,
        intrinsic_dimension=dimension,
        rigidity_rank=rigidity_rank,
        maximum_rigidity_rank=maximum_rank,
        rigidity_rank_fraction=rigidity_fraction,
        infinitesimally_rigid=maximum_rank > 0 and rigidity_rank == maximum_rank,
        received_payload_bytes=directed_edges * payload_bytes_per_neighbor,
    )
