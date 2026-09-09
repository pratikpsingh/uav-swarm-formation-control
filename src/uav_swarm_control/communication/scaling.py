"""Factorial design helpers for swarm-size and neighbor-count research."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NeighborScalingCondition:
    swarm_size: int
    requested_neighbors: int

    def __post_init__(self) -> None:
        if self.swarm_size < 2:
            raise ValueError("swarm_size must be at least two.")
        if not 0 <= self.requested_neighbors < self.swarm_size:
            raise ValueError("requested_neighbors must lie in [0, swarm_size - 1].")

    @property
    def normalized_degree(self) -> float:
        """Requested outgoing-neighbor fraction relative to all-to-all exchange."""
        return self.requested_neighbors / (self.swarm_size - 1)

    @property
    def requested_directed_edges(self) -> int:
        """Directed messages per update when range does not remove neighbors."""
        return self.swarm_size * self.requested_neighbors

    @property
    def all_to_all_directed_edges(self) -> int:
        return self.swarm_size * (self.swarm_size - 1)

    @property
    def communication_reduction_fraction(self) -> float:
        """Ideal payload reduction versus all-to-all exchange, excluding headers."""
        return 1.0 - self.normalized_degree

    @property
    def minimum_generic_3d_rigidity_edges(self) -> int:
        """Maxwell's necessary generic edge count in 3D for N >= 3."""
        return max(0, 3 * self.swarm_size - 6)

    @property
    def generic_3d_rigidity_average_degree_bound(self) -> float:
        """Necessary undirected mean degree from the 3N - 6 edge count."""
        return 6.0 - 12.0 / self.swarm_size

    @property
    def possible_undirected_edge_bounds(self) -> tuple[int, int]:
        """Bounds after symmetrizing a range-unlimited directed top-k graph.

        Reciprocal directed selections share one undirected edge, while a
        non-reciprocal selection may create a distinct edge. Geometry decides
        the realized value inside these bounds.
        """
        directed = self.requested_directed_edges
        maximum_possible = self.swarm_size * (self.swarm_size - 1) // 2
        return math.ceil(directed / 2), min(directed, maximum_possible)

    @property
    def can_meet_generic_3d_edge_count(self) -> bool:
        """Whether edge count could meet the necessary bound, never rigidity itself."""
        return self.possible_undirected_edge_bounds[1] >= self.minimum_generic_3d_rigidity_edges

    @property
    def meets_count_bound_under_full_reciprocity(self) -> bool:
        """Conservative count check if every directed selection is reciprocal."""
        return self.possible_undirected_edge_bounds[0] >= self.minimum_generic_3d_rigidity_edges


def neighbor_scaling_grid(
    swarm_sizes: tuple[int, ...],
    requested_counts: tuple[int, ...],
    *,
    include_all: bool = True,
) -> tuple[NeighborScalingCondition, ...]:
    """Build unique valid (N,k) conditions, optionally including every all-neighbor case."""
    conditions: list[NeighborScalingCondition] = []
    for size in swarm_sizes:
        counts = set(requested_counts)
        if include_all:
            counts.add(size - 1)
        conditions.extend(
            NeighborScalingCondition(size, count) for count in sorted(counts) if 0 <= count < size
        )
    return tuple(conditions)


__all__ = ["NeighborScalingCondition", "neighbor_scaling_grid"]
