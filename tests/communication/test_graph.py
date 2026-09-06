"""Tests for directed neighbor selection and graph/rigidity descriptors."""

import numpy as np

from uav_swarm_control.communication import (
    build_neighbor_adjacency,
    communication_graph_metrics,
)


def test_complete_square_is_connected_and_intrinsically_rigid() -> None:
    positions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    metrics = communication_graph_metrics(
        positions,
        requested_neighbors=3,
        sensing_radius_m=None,
        payload_bytes_per_neighbor=24,
    )

    assert metrics.directed_edge_count == 12
    assert metrics.actual_degree_mean == 3.0
    assert metrics.connected
    assert np.isclose(metrics.algebraic_connectivity, 4.0)
    assert metrics.intrinsic_dimension == 2
    assert metrics.rigidity_rank == metrics.maximum_rigidity_rank == 5
    assert metrics.infinitesimally_rigid
    assert metrics.received_payload_bytes == 288


def test_range_can_reduce_actual_degree_below_requested_degree() -> None:
    positions = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [3.0, 0.0, 1.0], [3.5, 0.0, 1.0]])

    adjacency = build_neighbor_adjacency(
        positions,
        requested_neighbors=3,
        sensing_radius_m=0.75,
    )
    metrics = communication_graph_metrics(
        positions,
        requested_neighbors=3,
        sensing_radius_m=0.75,
        payload_bytes_per_neighbor=24,
    )

    assert adjacency.sum(axis=1).tolist() == [1, 1, 1, 1]
    assert metrics.actual_degree_mean == 1.0
    assert metrics.connected_components == 2
    assert not metrics.connected
    assert metrics.algebraic_connectivity == 0.0


def test_zero_neighbors_has_zero_connectivity_rigidity_and_bytes() -> None:
    positions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    metrics = communication_graph_metrics(
        positions,
        requested_neighbors=0,
        sensing_radius_m=None,
        payload_bytes_per_neighbor=24,
    )

    assert metrics.connected_components == 3
    assert metrics.rigidity_rank == 0
    assert metrics.received_payload_bytes == 0
