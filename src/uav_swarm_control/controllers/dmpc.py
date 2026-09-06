"""Finite-horizon DMPC adapted from the native DMPC-Swarm trajectory formulation."""

import math
from collections.abc import Mapping
from time import perf_counter
from typing import cast

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, minimize

from uav_swarm_control.configuration import DMPCConfig
from uav_swarm_control.environments.contracts import NormalizedVelocityActions
from uav_swarm_control.formations._typing import FloatArray
from uav_swarm_control.observations import CentralizedState, LocalObservations

type FloatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
type FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def triple_integrator_matrices(time_step_seconds: float) -> tuple[FloatArray, FloatArray]:
    """Return exact zero-order-hold matrices for position, velocity, acceleration, and jerk."""
    if not math.isfinite(time_step_seconds) or time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be finite and positive.")
    dt = time_step_seconds
    axis_state = np.array(
        [[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    axis_input = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]], dtype=np.float64)
    return (
        np.asarray(np.kron(axis_state, np.eye(3)), dtype=np.float64),
        np.asarray(np.kron(axis_input, np.eye(3)), dtype=np.float64),
    )


def prediction_matrices(horizon: int, time_step_seconds: float) -> tuple[FloatArray, FloatArray]:
    """Stack x[1] ... x[H] = free @ x[0] + control @ [u[0] ... u[H-1]]."""
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    state_matrix, input_matrix = triple_integrator_matrices(time_step_seconds)
    free = np.zeros((9 * horizon, 9), dtype=np.float64)
    control = np.zeros((9 * horizon, 3 * horizon), dtype=np.float64)
    for step in range(horizon):
        free[9 * step : 9 * (step + 1)] = np.linalg.matrix_power(state_matrix, step + 1)
        for control_step in range(step + 1):
            control[
                9 * step : 9 * (step + 1),
                3 * control_step : 3 * (control_step + 1),
            ] = np.linalg.matrix_power(state_matrix, step - control_step) @ input_matrix
    return free, control


def _rows(horizon: int, offset: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(9 * step + offset, 9 * step + offset + 3) for step in range(horizon)]
    )


class DistributedMPCController:
    """Solve one constrained trajectory QP per agent from a common state snapshot.

    The native repository distributes state and planned trajectories over a communication
    protocol. This in-process adapter supplies the same kind of snapshot directly, then solves
    independent per-agent problems. It is therefore a controller baseline, not a communication
    benchmark or a bit-for-bit execution of the original software.
    """

    def __init__(
        self,
        config: DMPCConfig,
        *,
        control_time_step_seconds: float,
        max_velocity_component_mps: float,
        max_neighbors: int,
        neighbor_radius_m: float | None,
    ) -> None:
        if not math.isfinite(control_time_step_seconds) or control_time_step_seconds <= 0.0:
            raise ValueError("control_time_step_seconds must be finite and positive.")
        if not math.isfinite(max_velocity_component_mps) or max_velocity_component_mps <= 0.0:
            raise ValueError("max_velocity_component_mps must be finite and positive.")
        ratio = config.planning_interval_seconds / control_time_step_seconds
        planning_steps = round(ratio)
        if planning_steps < 1 or not math.isclose(ratio, planning_steps, abs_tol=1e-10):
            raise ValueError(
                "planning interval must be an integer multiple of the control timestep."
            )
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be a non-negative integer.")
        if neighbor_radius_m is not None and (
            not math.isfinite(neighbor_radius_m) or neighbor_radius_m <= 0.0
        ):
            raise ValueError("neighbor_radius_m must be finite and positive when set.")

        self.config = config
        self.control_time_step_seconds = control_time_step_seconds
        self.max_velocity_component_mps = max_velocity_component_mps
        self.max_neighbors = max_neighbors
        self.neighbor_radius_m = neighbor_radius_m
        self._planning_steps = planning_steps
        self._free, self._control = prediction_matrices(
            config.prediction_horizon_steps, config.planning_interval_seconds
        )
        position_rows = _rows(config.prediction_horizon_steps, 0)
        velocity_rows = _rows(config.prediction_horizon_steps, 3)
        acceleration_rows = _rows(config.prediction_horizon_steps, 6)
        self._free_position = self._free[position_rows]
        self._free_velocity = self._free[velocity_rows]
        self._free_acceleration = self._free[acceleration_rows]
        self._control_position = self._control[position_rows]
        self._control_velocity = self._control[velocity_rows]
        self._control_acceleration = self._control[acceleration_rows]
        self._terminal_position_control = self._control_position[-3:]
        weights = config.weights
        dimension = 3 * config.prediction_horizon_steps
        self._hessian = 2.0 * (
            weights.terminal_position
            * self._terminal_position_control.T
            @ self._terminal_position_control
            + weights.velocity * self._control_velocity.T @ self._control_velocity
            + weights.acceleration * self._control_acceleration.T @ self._control_acceleration
            + weights.jerk * np.eye(dimension)
        )
        self._regularized_hessian = self._hessian + np.eye(dimension) * 1e-10
        self.reset()

    def reset(self) -> None:
        """Clear warm starts, state estimates, and episode-local diagnostics."""
        self._plans: FloatArray | None = None
        self._plan_initial_states: FloatArray | None = None
        self._previous_replan_velocities: FloatArray | None = None
        self._phase = self._planning_steps
        self._replans = 0
        self._solve_seconds: list[float] = []
        self._objective_values: list[float] = []
        self._iterations: list[float] = []
        self._active_constraints: list[float] = []
        self._predicted_separations: list[float] = []
        self._successful_solves = 0
        self._failed_solves = 0

    def act_with_state(
        self,
        observations: LocalObservations,
        state: CentralizedState,
    ) -> NormalizedVelocityActions:
        """Replan at the native 5 Hz cadence and emit the next 30 Hz velocity setpoint."""
        count = observations.num_agents
        if self.max_neighbors > count - 1:
            raise ValueError("max_neighbors cannot exceed num_agents - 1.")
        values = np.asarray(state.values, dtype=np.float64)
        if values.shape != (count * 9,):
            raise ValueError(
                f"centralized state must contain positions, velocities, targets for {count} agents."
            )
        positions = values[: 3 * count].reshape(count, 3)
        velocities = values[3 * count : 6 * count].reshape(count, 3)
        targets = values[6 * count :].reshape(count, 3)
        if self._plans is None or self._phase >= self._planning_steps:
            accelerations = self._estimate_accelerations(velocities)
            self._plans = self._replan(positions, velocities, accelerations, targets)
            self._plan_initial_states = np.concatenate(
                (positions, velocities, accelerations), axis=1
            )
            self._previous_replan_velocities = velocities.copy()
            self._phase = 0

        if self._plan_initial_states is None:
            raise RuntimeError("DMPC plan was not initialized.")
        elapsed = (self._phase + 1) * self.control_time_step_seconds
        commands = np.vstack(
            [
                self._planned_velocity(
                    self._plan_initial_states[index], self._plans[index], elapsed
                )
                for index in range(count)
            ]
        )
        self._phase += 1
        normalized = np.clip(commands / self.max_velocity_component_mps, -1.0, 1.0)
        return NormalizedVelocityActions(observations.agent_ids, normalized.astype(np.float32))

    def diagnostics(self) -> Mapping[str, float]:
        """Return optimization diagnostics that must not be mixed with task metrics."""
        runs = len(self._solve_seconds)
        return {
            "replans": float(self._replans),
            "optimization_runs": float(runs),
            "successful_solves": float(self._successful_solves),
            "failed_solves": float(self._failed_solves),
            "solver_success_rate": self._successful_solves / runs if runs else 0.0,
            "mean_solve_time_ms": 1000.0 * float(np.mean(self._solve_seconds)) if runs else 0.0,
            "max_solve_time_ms": 1000.0 * max(self._solve_seconds, default=0.0),
            "mean_solver_iterations": float(np.mean(self._iterations)) if runs else 0.0,
            "mean_objective": float(np.mean(self._objective_values)) if runs else 0.0,
            "mean_active_collision_constraints": (
                float(np.mean(self._active_constraints)) if runs else 0.0
            ),
            "minimum_predicted_scaled_separation_m": min(self._predicted_separations, default=0.0),
        }

    def _estimate_accelerations(self, velocities: FloatArray) -> FloatArray:
        if (
            self._previous_replan_velocities is None
            or self._previous_replan_velocities.shape != velocities.shape
        ):
            return np.zeros_like(velocities)
        estimate = (
            velocities - self._previous_replan_velocities
        ) / self.config.planning_interval_seconds
        limit = self.config.limits.acceleration_mps2
        return np.clip(estimate, -limit, limit)

    def _replan(
        self,
        positions: FloatArray,
        velocities: FloatArray,
        accelerations: FloatArray,
        targets: FloatArray,
    ) -> FloatArray:
        states = np.concatenate((positions, velocities, accelerations), axis=1)
        nominal = np.stack(
            [self._nominal_plan(states[index], targets[index]) for index in range(len(states))]
        )
        nominal_positions = np.stack(
            [self._predict(plan, states[index])[0] for index, plan in enumerate(nominal)]
        )
        plans = np.empty_like(nominal)
        for index in range(len(states)):
            warm = self._warm_start(index, nominal[index])
            plans[index] = self._solve_agent(
                index,
                states[index],
                targets[index],
                positions,
                nominal_positions,
                warm,
            )
        self._replans += 1
        return plans

    def _nominal_plan(self, state: FloatArray, target: FloatArray) -> FloatArray:
        gradient = self._gradient(state, target)
        solution = np.linalg.solve(self._regularized_hessian, -gradient)
        limit = self.config.limits.jerk_mps3
        return np.clip(solution, -limit, limit).reshape(self.config.prediction_horizon_steps, 3)

    def _warm_start(self, index: int, nominal: FloatArray) -> FloatArray:
        if self._plans is None or index >= len(self._plans):
            return nominal
        previous = self._plans[index]
        return np.vstack((previous[1:], previous[-1]))

    def _gradient(self, state: FloatArray, target: FloatArray) -> FloatArray:
        weights = self.config.weights
        free_terminal = self._free_position[-3:] @ state
        return 2.0 * (
            weights.terminal_position * self._terminal_position_control.T @ (free_terminal - target)
            + weights.velocity * self._control_velocity.T @ (self._free_velocity @ state)
            + weights.acceleration
            * self._control_acceleration.T
            @ (self._free_acceleration @ state)
        )

    def _solve_agent(
        self,
        index: int,
        state: FloatArray,
        target: FloatArray,
        positions: FloatArray,
        nominal_positions: FloatArray,
        warm: FloatArray,
    ) -> FloatArray:
        gradient = self._gradient(state, target)
        constraints: list[LinearConstraint] = [
            self._symmetric_constraint(
                self._control_velocity,
                self._free_velocity @ state,
                self.max_velocity_component_mps,
            ),
            self._symmetric_constraint(
                self._control_acceleration,
                self._free_acceleration @ state,
                self.config.limits.acceleration_mps2,
            ),
        ]
        collision_matrix, collision_bound = self._collision_constraints(
            index, state, positions, nominal_positions
        )
        if len(collision_bound):
            constraints.append(LinearConstraint(collision_matrix, -np.inf, collision_bound))
        flat_warm = warm.reshape(-1)

        def objective(control: FloatVector) -> float:
            return float(0.5 * control @ self._hessian @ control + gradient @ control)

        def jacobian(control: FloatVector) -> FloatVector:
            return cast(FloatVector, self._hessian @ control + gradient)

        started = perf_counter()
        result: OptimizeResult[np.float64] = minimize(
            objective,
            flat_warm,
            jac=jacobian,
            method="SLSQP",
            bounds=Bounds(
                -self.config.limits.jerk_mps3,
                self.config.limits.jerk_mps3,
            ),
            constraints=constraints,
            options={
                "maxiter": self.config.solver.max_iterations,
                "ftol": self.config.solver.function_tolerance,
                "disp": False,
            },
        )
        solve_seconds = perf_counter() - started
        candidate = np.asarray(result.x, dtype=np.float64)
        feasible = self._is_feasible(candidate, state, collision_matrix, collision_bound)
        success = bool(result.success and feasible and np.isfinite(candidate).all())
        chosen = candidate if success else flat_warm
        self._successful_solves += int(success)
        self._failed_solves += int(not success)
        self._solve_seconds.append(solve_seconds)
        self._iterations.append(float(getattr(result, "nit", 0)))
        self._objective_values.append(self._trajectory_cost(chosen, state, target))
        if len(collision_bound):
            slack = collision_bound - collision_matrix @ chosen
            tolerance = self.config.solver.constraint_tolerance
            self._active_constraints.append(float(np.count_nonzero(slack <= tolerance)))
        else:
            self._active_constraints.append(0.0)
        predicted, _, _ = self._predict(chosen.reshape(-1, 3), state)
        self._predicted_separations.append(
            self._minimum_scaled_separation(index, predicted, nominal_positions)
        )
        return chosen.reshape(self.config.prediction_horizon_steps, 3)

    @staticmethod
    def _symmetric_constraint(
        matrix: FloatArray, free: FloatArray, limit: float
    ) -> LinearConstraint:
        return LinearConstraint(
            cast(FloatMatrix, matrix),
            cast(FloatVector, -limit - free),
            cast(FloatVector, limit - free),
        )

    def _collision_constraints(
        self,
        index: int,
        state: FloatArray,
        positions: FloatArray,
        nominal_positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        neighbors = self._neighbor_indices(index, positions)
        horizon = self.config.prediction_horizon_steps
        scaling = np.diag([1.0, 1.0, 1.0 / self.config.limits.downwash_scaling])
        free_positions = (self._free_position @ state).reshape(horizon, 3)
        rows: list[FloatArray] = []
        bounds: list[float] = []
        for other in neighbors:
            for step in range(horizon):
                difference = scaling @ (
                    nominal_positions[index, step] - nominal_positions[other, step]
                )
                distance = math.sqrt(sum(float(value) ** 2 for value in difference))
                if distance <= 1e-12:
                    difference = np.zeros(3, dtype=np.float64)
                    difference[(index + other) % 3] = 1.0 if index < other else -1.0
                    distance = 1.0
                normal = difference / distance
                midpoint = 0.5 * (nominal_positions[index, step] + nominal_positions[other, step])
                control_block = self._control_position[3 * step : 3 * (step + 1)]
                rows.append(-(normal @ scaling @ control_block))
                bounds.append(
                    -0.5 * self.config.limits.minimum_separation_m
                    + float(normal @ scaling @ (free_positions[step] - midpoint))
                )
        if not rows:
            return np.empty((0, self._control.shape[1]), dtype=np.float64), np.empty(
                (0,), dtype=np.float64
            )
        return np.vstack(rows), np.asarray(bounds, dtype=np.float64)

    def _neighbor_indices(self, index: int, positions: FloatArray) -> list[int]:
        distances = [
            math.sqrt(
                sum(float(component) ** 2 for component in positions[other] - positions[index])
            )
            for other in range(len(positions))
        ]
        candidates = [
            other
            for other in range(len(positions))
            if other != index
            and (self.neighbor_radius_m is None or distances[other] <= self.neighbor_radius_m)
        ]
        candidates.sort(key=lambda other: (round(float(distances[other]), 12), other))
        return candidates[: self.max_neighbors]

    def _is_feasible(
        self,
        candidate: FloatArray,
        state: FloatArray,
        collision_matrix: FloatArray,
        collision_bound: FloatArray,
    ) -> bool:
        tolerance = self.config.solver.constraint_tolerance
        if np.any(np.abs(candidate) > self.config.limits.jerk_mps3 + tolerance):
            return False
        velocity = self._free_velocity @ state + self._control_velocity @ candidate
        acceleration = self._free_acceleration @ state + self._control_acceleration @ candidate
        collision_ok = not len(collision_bound) or np.all(
            collision_matrix @ candidate <= collision_bound + tolerance
        )
        return bool(
            np.all(np.abs(velocity) <= self.max_velocity_component_mps + tolerance)
            and np.all(np.abs(acceleration) <= self.config.limits.acceleration_mps2 + tolerance)
            and collision_ok
        )

    def _predict(
        self, plan: FloatArray, state: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        control = plan.reshape(-1)
        positions = (self._free_position @ state + self._control_position @ control).reshape(-1, 3)
        velocities = (self._free_velocity @ state + self._control_velocity @ control).reshape(-1, 3)
        accelerations = (
            self._free_acceleration @ state + self._control_acceleration @ control
        ).reshape(-1, 3)
        return positions, velocities, accelerations

    def _trajectory_cost(self, control: FloatArray, state: FloatArray, target: FloatArray) -> float:
        positions, velocities, accelerations = self._predict(control.reshape(-1, 3), state)
        weights = self.config.weights
        return (
            weights.terminal_position * self._sum_squares(positions[-1] - target)
            + weights.velocity * self._sum_squares(velocities)
            + weights.acceleration * self._sum_squares(accelerations)
            + weights.jerk * self._sum_squares(control)
        )

    def _minimum_scaled_separation(
        self, index: int, predicted: FloatArray, nominal_positions: FloatArray
    ) -> float:
        neighbors = [other for other in range(len(nominal_positions)) if other != index]
        if not neighbors:
            return 0.0
        scaling = np.array([1.0, 1.0, 1.0 / self.config.limits.downwash_scaling], dtype=np.float64)
        return min(
            min(
                math.sqrt(sum(float(component) ** 2 for component in row))
                for row in (predicted - nominal_positions[other]) * scaling
            )
            for other in neighbors
        )

    @staticmethod
    def _sum_squares(values: FloatArray) -> float:
        return sum(float(value) ** 2 for value in values.reshape(-1))

    def _planned_velocity(
        self, initial_state: FloatArray, plan: FloatArray, elapsed: float
    ) -> FloatArray:
        state = initial_state.copy()
        remaining = elapsed
        interval = self.config.planning_interval_seconds
        for jerk in plan:
            duration = min(remaining, interval)
            if duration <= 0.0:
                break
            position = state[:3].copy()
            velocity = state[3:6].copy()
            acceleration = state[6:].copy()
            state[:3] = (
                position
                + duration * velocity
                + 0.5 * duration**2 * acceleration
                + duration**3 * jerk / 6.0
            )
            state[3:6] = velocity + duration * acceleration + 0.5 * duration**2 * jerk
            state[6:] = acceleration + duration * jerk
            remaining -= duration
        return np.clip(
            state[3:6],
            -self.max_velocity_component_mps,
            self.max_velocity_component_mps,
        )
