# ADR 0007: Isolate and pin the PyBullet UAV simulator

- Status: accepted
- Date: 2026-09-05

## Context

Paper comparisons require more realistic dynamics than the point-mass environment, but directly
coupling task and learning code to a simulator would make metrics, action meaning, and future
simulator changes difficult to audit. The upstream velocity aviary also defines a specialized
direction-and-speed action rather than this project's three physical velocity components.

## Decision

- Pin gym-pybullet-drones 2.2.0 to Git revision
  `24fccff4da746badb6eaa9f5055cd5d9b6613160` and verify it at runtime.
- Wrap `CtrlAviary` and use one `DSLPIDControl` per UAV.
- Preserve the project action contract: each normalized component scales to a world-frame m/s
  target.
- Put simulator state behind the typed `DronePhysicsBackend` protocol.
- Reuse simulator-independent observation, reward, termination, and metric functions.
- Enforce an integral physics/control-rate ratio and a matching environment timestep.
- Reset simulator and PID memory together.
- Save resolved configuration and simulator provenance for every CLI run.
- Keep the simulator in an optional `simulation` extra and do not vendor upstream code.

## Consequences

The kinematic and PyBullet tasks are comparable at the interface and metric levels while their
dynamics remain honestly different. Unit tests can use a tiny fake backend, while two integration
tests prove real hover and formation flight. The simulator is training/evaluation infrastructure,
not a deployment dependency. Upstream internals are isolated to one bridge module, but changes to
the pinned revision require a deliberate audit and adapter update.
