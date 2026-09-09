"""Tests for neighbor-count factorial design helpers."""

from uav_swarm_control.communication.scaling import neighbor_scaling_grid


def test_neighbor_grid_omits_invalid_counts_and_adds_all_neighbor_case() -> None:
    grid = neighbor_scaling_grid((4, 8), (0, 1, 2, 4, 8))
    assert [(item.swarm_size, item.requested_neighbors) for item in grid] == [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 7),
    ]
    assert grid[-1].normalized_degree == 1.0


def test_neighbor_condition_separates_payload_and_rigidity_bounds() -> None:
    condition = neighbor_scaling_grid((8,), (2,), include_all=False)[0]

    assert condition.requested_directed_edges == 16
    assert condition.all_to_all_directed_edges == 56
    assert condition.communication_reduction_fraction == 5.0 / 7.0
    assert condition.minimum_generic_3d_rigidity_edges == 18
    assert condition.generic_3d_rigidity_average_degree_bound == 4.5
    assert condition.possible_undirected_edge_bounds == (8, 16)
    assert not condition.can_meet_generic_3d_edge_count
    assert not condition.meets_count_bound_under_full_reciprocity

    denser = neighbor_scaling_grid((8,), (5,), include_all=False)[0]
    assert denser.possible_undirected_edge_bounds == (20, 28)
    assert denser.can_meet_generic_3d_edge_count
    assert denser.meets_count_bound_under_full_reciprocity
