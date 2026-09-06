"""Deterministic fixed and minimum-cost formation assignment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.formations._validation import matching_point_arrays


@dataclass(frozen=True, slots=True)
class FormationAssignment:
    """Target row assigned to each agent row and its total Euclidean travel distance."""

    target_indices: tuple[int, ...]
    assigned_targets: FloatArray
    total_distance_m: float


def fixed_assignment(origins: ArrayLike, targets: ArrayLike) -> FormationAssignment:
    """Preserve row correspondence between agents and target points."""
    origin_array, target_array = matching_point_arrays(origins, targets)
    indices = tuple(range(len(origin_array)))
    assigned = target_array.copy()
    return FormationAssignment(
        indices,
        assigned,
        float(np.linalg.norm(origin_array - assigned, axis=1).sum()),
    )


def _hungarian(costs: FloatArray) -> tuple[int, ...]:
    """Solve a square linear assignment in O(N^3), breaking ties by target index."""
    count = costs.shape[0]
    row_potential = np.zeros(count + 1, dtype=np.float64)
    column_potential = np.zeros(count + 1, dtype=np.float64)
    matched_row = np.zeros(count + 1, dtype=np.int64)
    predecessor = np.zeros(count + 1, dtype=np.int64)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.abs(costs).max())) * 16

    for row in range(1, count + 1):
        matched_row[0] = row
        minimum = np.full(count + 1, np.inf, dtype=np.float64)
        used = np.zeros(count + 1, dtype=np.bool_)
        column = 0
        while True:
            used[column] = True
            active_row = int(matched_row[column])
            delta = np.inf
            next_column = 0
            for candidate in range(1, count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1, candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate] - tolerance:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta - tolerance or (
                    abs(minimum[candidate] - delta) <= tolerance
                    and (next_column == 0 or candidate < next_column)
                ):
                    delta = minimum[candidate]
                    next_column = candidate
            if not np.isfinite(delta):
                raise ValueError("assignment costs do not admit a finite complete matching.")
            for candidate in range(count + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = int(predecessor[column])
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = np.empty(count, dtype=np.int64)
    for column in range(1, count + 1):
        assignment[matched_row[column] - 1] = column - 1
    return tuple(int(index) for index in assignment)


def minimum_distance_assignment(
    origins: ArrayLike,
    targets: ArrayLike,
) -> FormationAssignment:
    """Minimize the sum of agent-to-target Euclidean distances without SciPy."""
    origin_array, target_array = matching_point_arrays(origins, targets)
    differences = origin_array[:, None, :] - target_array[None, :, :]
    costs = np.linalg.norm(differences, axis=2)
    indices = _hungarian(costs)
    assigned = target_array[np.asarray(indices, dtype=np.int64)].copy()
    return FormationAssignment(
        indices,
        assigned,
        float(sum(costs[row, column] for row, column in enumerate(indices))),
    )


__all__ = ["FormationAssignment", "fixed_assignment", "minimum_distance_assignment"]
