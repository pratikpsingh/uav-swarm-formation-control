# Classical DMPC baseline

The implemented method is `dmpc-swarm-clean-room-adaptation`. It preserves the native
DMPC-Swarm trajectory model and repository controller constants where they can be identified, but it
is not a bit-for-bit execution of `DMPC-to-MARL-Sim`. Its purpose is a controlled classical baseline
inside the same environment and evaluator used by the corrected Paper 04 MAPPO baseline.

## Source audit and method identity

The sibling `DMPC-to-MARL-Sim` repository was inspected at revision
`9ee543d9e2e553708f09f57c7ebc2f55f0302df8`. Its `run_dampc=True` branch selects the learned
approximate-MPC path; it must not be reported as classical DMPC. The classical path constructs a
finite-horizon trajectory with `TrajectoryGenerator.calculate_trajectory()`.

| Property | Native repository | This baseline |
| --- | --- | --- |
| Motion model | 3D triple integrator; state is position, velocity, acceleration; input is jerk | Same state and exact discrete-time model |
| Planning cadence | 5 Hz (`dt = 0.2 s`) | Same; velocity commands are held/interpolated through the 30 Hz control loop |
| Prediction horizon | 15 steps / 3 seconds | Same |
| Costs | terminal position 1, velocity 0.0001, acceleration 0.005, jerk 0.001 | Same explicit YAML values |
| Limits | acceleration 5, jerk 5, minimum separation 0.4 m | Same; shared task supplies velocity limit |
| Downwash geometry | vertical scale 4 | Same ellipsoidal-distance scaling |
| Numerical solver | `qpsolvers` with Quadprog | SciPy SLSQP with explicit convergence/feasibility checks |
| Neighbor information | communicated plans, priorities, delays and loss options | synchronous in-process positions, velocities and assigned targets |
| Collision handling | buffered Voronoi constraints using communicated trajectories | linearized separation half-spaces around simultaneous nominal plans |
| Deadlock handling | high-level planner | not implemented |
| Execution | asynchronous distributed compute units | one independent optimization per UAV from one shared snapshot |

No source from the sibling repository is imported or vendored. The result manifest records both the
method label and `exact_native_software_execution: false` so downstream reports cannot silently
upgrade this adaptation into a reproduction claim.

## Controller model

For each Cartesian axis, the planner uses position `p`, velocity `v`, acceleration `a`, and jerk
input `j`. With planning interval `dt`:

```text
p[k+1] = p[k] + dt v[k] + 0.5 dt^2 a[k] + (dt^3 / 6) j[k]
v[k+1] = v[k] + dt a[k] + 0.5 dt^2 j[k]
a[k+1] = a[k] + dt j[k]
```

The three axes are stacked into one linear system. Prediction matrices express the entire horizon
as `X = A_bar x0 + B_bar J`, so the terminal-position, velocity, acceleration and jerk penalties
form a quadratic objective. Every replan estimates current acceleration from velocity changes,
builds an unconstrained nominal trajectory for every UAV, then solves one constrained problem per
UAV. Only the first part of each optimized trajectory is executed; after 0.2 seconds, the state is
measured and the problem is solved again. This repeated correction is the receding-horizon idea.

Jerk bounds use box constraints. Velocity and acceleration bounds, plus linearized collision
half-spaces against selected neighbors, use linear constraints. A solution is accepted only when
the optimizer reports success and the returned values independently satisfy the configured
tolerance. A rejected solution falls back to a shifted previous plan, or the bounded nominal plan
on the first replan. Solver failure is therefore observable in diagnostics instead of crashing the
episode or being silently called success.

## What is fair—and what is not

MAPPO and DMPC share the Stage 7 `Paper04Environment`, task YAML, initial-state seeds, CF2X physics,
30/240 Hz timing, action scaling, episode horizon, termination rules, and trajectory metrics. A
comparison fingerprint covers the full physics configuration, evaluation root seed, episode count,
horizon, and metric-schema version. `compare-controllers` refuses different fingerprints.

The information sets are intentionally different. The deployed MAPPO actor receives its declared
local observation. This DMPC adapter receives shared positions, velocities, assigned targets and
stable agent identities so it can model neighbor trajectories. The current comparison therefore
answers “how do these controllers perform on the same physical task?” It does not answer “which
method is better under equal communication bandwidth?” Stage 11 must make communication access,
delay, loss and bytes explicit before that claim is possible.

Common task metrics are kept separate from controller internals. Common metrics include success,
collisions, clearance, position/shape error, path length, action smoothness and episode return.
DMPC-only diagnostics include optimization count, solver success rate, solve time, iterations,
objective value, active collision constraints and predicted scaled separation. Reward and DMPC
objective values are not interchangeable performance measures.

MAPPO variability is computed across independently trained policies. Deterministic DMPC has no
training seeds, so its variability is across evaluation initial conditions. These are different
sampling units. The comparison artifact names both explicitly and makes no significance claim.

## Install and run

Development environments already include the required packages. For a runtime installation:

```bash
uv sync --no-dev --extra simulation --extra dmpc
```

Run the bounded compatible smoke profile:

```bash
uv run uav-swarm-control run-dmpc \
  --task configs/experiment/paper04_3uav.yaml \
  --task configs/experiment/paper04_4uav.yaml \
  --task configs/experiment/paper04_5uav.yaml \
  --controller configs/algorithm/dmpc_native.yaml \
  --smoke \
  --output artifacts/baselines/dmpc-check \
  --project-root .
```

Smoke mode applies exactly the same two-episode, 48-control-step evaluation profile as the MAPPO
smoke run. It is too short to assess task success. Omit `--smoke` for the configured 20 held-out
episodes and 242-step horizon. DMPC requires evaluation only; it has no training budget.

After producing a MAPPO summary under the same profile, create a guarded comparison:

```bash
uv run uav-swarm-control compare-controllers \
  --mappo-summary artifacts/baselines/paper04-v1/research/paper04-3uav/summary.json \
  --dmpc-result artifacts/baselines/dmpc-v1/research/paper04-3uav/dmpc-swarm-reference/result.json \
  --output artifacts/baselines/comparisons/paper04-3uav.json
```

Use different output roots for smoke and research work. Existing DMPC result directories and
comparison files are never overwritten.

## Artifacts and limitations

Each controller directory contains `manifest.json`, `source.zip`, and `result.json`. The manifest
saves the resolved task/controller configuration, source and dependency provenance, native
reference revision, method identity, and comparison protocol. The result contains raw common
episodes, their summary, raw controller diagnostics, their separate summary, simulator metadata,
information access and evaluation wall time.

The current adapter does not reproduce the native asynchronous communication scheduler, stale
neighbor plans, priority/event-triggering protocol, quantization, packet loss, network delay, the
high-level deadlock planner, or Quadprog numerics. Its collision constraints use one-shot nominal
neighbor trajectories rather than a distributed fixed-point exchange. Symmetric crossing tasks can
therefore deadlock or make the local linearized problem infeasible. Those behaviors must be treated
as limitations to test, not hidden by retuning on evaluation results.

Run `uv run pytest tests/controllers/test_dmpc.py tests/evaluation/test_comparison.py -q` for the
focused mathematical/protocol checks. The repository gate remains Ruff format/lint, Pyright,
`uv run pytest --cov`, and `uv build`.
