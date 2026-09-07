# ADR 0009: Compare a clean-room native-formulation DMPC baseline

- Status: accepted
- Date: 2026-09-06

## Context

`DMPC-to-MARL-Sim` contains both a classical trajectory optimizer and a learned approximate-MPC
path. Its `run_dampc` name refers to the learned path, so selecting it would not produce the
classical baseline required by the roadmap. Directly embedding the old asynchronous simulator and
network stack would also prevent a controlled comparison with the MAPPO baseline task.

## Decision

Implement the native triple-integrator, horizon and published controller constants cleanly behind
a state-aware controller protocol. Solve one finite-horizon constrained problem per UAV at 5 Hz,
then adapt its predicted velocity to the existing normalized 30 Hz action contract. Use the same
PyBullet environment, initial-condition seeds, termination rules and trajectory evaluator as
MAPPO.

Make every controller choice a strict YAML value. Archive source, resolved configuration,
dependency provenance and the audited native revision. Mark the implementation as an adaptation,
not exact native execution. Separate task metrics from optimization diagnostics and require a
fingerprint match before generating a MAPPO/DMPC comparison.

## Consequences

Classical and learned controllers can now be exercised through one physical benchmark without
changing either controller to know about the other. SciPy is an optional runtime dependency and is
loaded only for the DMPC command. The comparison controls environment conditions but not
information access: DMPC currently receives a synchronous shared snapshot while the MAPPO actor
uses local observations.

Native communication scheduling, stale plans, priorities, loss, quantization, deadlock resolution
and exact Quadprog behavior remain out of scope. A later communication experiment must add them as
independent, measured factors rather than implying they exist in this baseline.
