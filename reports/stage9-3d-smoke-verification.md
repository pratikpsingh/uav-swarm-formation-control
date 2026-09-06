# Stage 9 3D generalization smoke verification

Date: 2026-09-06. Profile: `smoke`. Method:
`stage9-3d-generalized-feedforward-mappo`.

The final-source physical smoke suite completed for plane, pyramid, cube, and sphere. It exercised
seven assignment/frame variants, five independent training seeds per variant, and two fresh episodes
from each of the in-distribution and held-out pose ranges. In total it produced 35 checkpoints, 8,960
environment transitions, 62,720 agent samples, and 140 evaluation episodes. A second invocation with
`--resume` verified every manifest/checkpoint hash and skipped all completed seeds.

| Shape / variant | Success in / held | Final position RMSE m in / held | Final shape RMSE in / held | Held collision | Held action clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| plane / minimum-target | 0 / 0 | 1.168 / 2.024 | 0.069 / 0.115 | 0 | 0 |
| pyramid / minimum-target | 0 / 0 | 1.182 / 2.057 | 0.068 / 0.376 | 0 | 0 |
| cube / fixed-world | 0 / 0 | 1.340 / 2.172 | 0.347 / 0.188 | 0 | 0 |
| cube / minimum-world | 0 / 0 | 1.352 / 2.027 | 0.356 / 0.119 | 0 | 0 |
| cube / fixed-target | 0 / 0 | 1.287 / 2.151 | 0.302 / 0.143 | 0 | 0 |
| cube / minimum-target | 0 / 0 | 1.292 / 2.062 | 0.267 / 0.128 | 0 | 0 |
| sphere / minimum-target | 0 / 0 | 1.108 / 2.109 | 0.041 / 0.362 | 0 | 0 |

Values are means of five independently trained policy means, each based on two episodes. “Held action
clip” is the mean fraction of action components clipped after target-to-world rotation. It is zero for
these nearly untrained deterministic actors; the diagnostic remains required for full runs.

Summed measured training and evaluation wall times were 176.43 and 128.47 seconds on this host.
Artifacts share source hash
`37a10f3a662b2f866adb9ece6e17ff6130cea3ce666170ae946522f09196112e`, Python 3.12.3,
Torch 2.14.0+cu130, NumPy 2.5.2, PyBullet 3.2.7, SciPy 1.18.1, and one Torch CPU thread. The recorded
Git revision is the committed Stage 8 parent `a604e4176ab741d55384c10429aef19bde1fd9e7`; `git_dirty` is
true because Stage 9 had not yet been committed. Exact source files are stored in each ignored
artifact snapshot. Gymnasium emitted its known Box float64-to-float32 precision-cast warnings.

Repository verification passed Ruff formatting and lint, Pyright with zero errors, `uv lock
--check`, and the full suite with 215 tests, 85% branch-aware coverage, and 8 upstream Gymnasium
precision-cast warnings. `uv build` produced both the source distribution and wheel.

All success values are zero because 256 training transitions and a 48-step (1.6-second) horizon are
plumbing limits, not a learning budget. Held-out position error is therefore not evidence of a
generalization ranking, and the cube ablation must not be interpreted scientifically. The smoke run
only demonstrates that pose sampling, assignments, frame transforms, Crazyflie physics, MAPPO,
checkpoint reload, two-split metrics, aggregation, provenance, and resume complete coherently.

Generated records are under ignored
`artifacts/generalization/stage9-verification-final-v2/smoke/`. The scientific gate remains open until
35 full-budget policies are trained on the frozen research YAMLs and their learning curves,
trajectories, action clipping, collisions, absolute split performance, and across-seed uncertainty are
reviewed.
