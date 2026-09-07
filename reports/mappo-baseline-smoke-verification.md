# Feed-forward MAPPO baseline pipeline verification

Date: 2026-09-06. This report records a bounded software smoke test, not convergence results.
Current method identity: `feedforward-mappo-baseline`. The preserved smoke bundle predates the naming refactor and records the legacy identifier `corrected-paper04-feedforward-mappo`. Exact Paper 04 reproduction: false.

Software checks: Ruff formatting and lint passed; Pyright reported no errors; the full regression
suite completed with 179 passed, 8 warnings, and 89% overall coverage (515.36 seconds). The warnings
are Gymnasium Box-bound float64-to-float32 precision casts in the upstream simulator integration.
The checkpoint evaluation test reproduced its saved episode records exactly on this host.
Both source distribution and wheel built successfully with uv build. These checks do not guarantee
bitwise reproducibility across different hardware or simulator versions.

All three tasks completed five independent 256-transition training runs, saved checkpoints,
reloaded the final policies, and evaluated two common held-out episodes per policy. The evaluation
horizon was 48 control steps. Training and inference used CPU; physics used the pinned Crazyflie
simulator, CF2X, all configured aerodynamic effects, 240 Hz physics and 30 Hz control.

| UAVs | Training seeds | Transitions per seed | Agent samples per seed | Success rate | Mean final normalized shape RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 5 | 256 | 768 | 0.0 | 0.067290 |
| 4 | 5 | 256 | 1024 | 0.0 | 0.049368 |
| 5 | 5 | 256 | 1280 | 0.0 | 0.056596 |

The zero success rates do not establish a broken implementation or a converged baseline. The
training budget and shortened horizon are deliberately insufficient for a research claim.
No relative performance conclusion should be drawn from the shape-error columns.

A separate proportional-controller check used the actual research horizon (242 steps), two held-out
episodes per task, and the same 30/240 Hz timing and aerodynamic settings. All six episodes succeeded.
Mean episode lengths were 81.5, 82.0, and 83.0 steps for 3, 4, and 5 UAVs respectively, with positive
minimum clearance. This establishes task feasibility under a known controller; it is not evidence
that a 256-step learned policy has converged. The diagnostic record is
`artifacts/baselines/verification/scripted-research-settings.json`.

Generated records predate the naming refactor and remain in the ignored legacy path `artifacts/baselines/verification/smoke/paper04-{3,4,5}uav/`. They are not renamed because their manifests record the original experiment identities.
Each folder contains exact source hashes/snapshot, configuration, simulator provenance, per-seed
updates, checkpoints, individual episode metrics and a summary. Source was uncommitted during
verification, based on commit `4d4c95d`; `source.zip` and hashes identify the actual executed code.

Required research follow-up: execute the 10M/30M/60M-transition configurations on the lab machine
for all five seeds, inspect training curves and failure behavior, and review the resulting
across-seed reports. The full recurrent Paper 04 method remains a separate reproduction target.
