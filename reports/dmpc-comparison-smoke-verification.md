# DMPC comparison pipeline verification

Date: 2026-09-06. This report records a bounded software smoke test, not scientific controller
performance. DMPC method: `dmpc-swarm-clean-room-adaptation`. Exact native execution: false.

The final-source smoke run evaluated the clean-room DMPC adapter and newly trained feed-forward MAPPO
smoke policies on the same 3-, 4-, and 5-UAV PyBullet tasks. Each MAPPO row aggregates five
independent 256-transition training runs; each policy used the same two held-out evaluation
episodes. DMPC used those two episode seeds directly, with no training. All evaluations used a
48-step (1.6-second) horizon, CF2X dynamics, 240 Hz physics and 30 Hz control.

| UAVs | Final position RMSE: MAPPO / DMPC (m) | Final normalized shape RMSE: MAPPO / DMPC | Minimum clearance: MAPPO / DMPC (m) | Success: MAPPO / DMPC |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.888928 / 0.847331 | 0.067290 / 0.109507 | 0.672752 / 0.580132 | 0.0 / 0.0 |
| 4 | 0.889329 / 0.864299 | 0.049368 / 0.078167 | 0.676821 / 0.603128 | 0.0 / 0.0 |
| 5 | 0.889891 / 0.887948 | 0.056596 / 0.070890 | 0.676705 / 0.611055 | 0.0 / 0.0 |

All three guarded comparisons accepted their protocol fingerprints. Every DMPC optimization
reported success in these episodes. Mean per-optimization solve times were 22.39, 18.54 and
19.91 ms for 3, 4 and 5 UAVs on this laptop; these are host-specific diagnostics, not deployment
guarantees. No collision threshold was crossed.

Repository verification passed Ruff formatting and lint, Pyright with zero errors, and the full
suite with 194 tests, 87% overall coverage and 8 upstream Gymnasium precision-cast warnings. `uv
build` produced both the source distribution and wheel. MAPPO and DMPC manifests share source hash
`638566d3f8dc48eff18f4554a714d152c65df060f1d6a560fda7d5e8d2c1d99b`.

Both controllers had zero success because the smoke horizon ends after only 1.6 seconds while the
formation center begins roughly 1 m from its target. The MAPPO policies are also intentionally
under-trained. The numbers above only demonstrate that configuration, physics, seeding, controller
dispatch, metric collection, solver diagnostics, provenance and compatibility checks complete.
They do not rank the methods, establish convergence or reproduce either source paper.

Generated records predate the naming refactor and remain under the ignored legacy path
`artifacts/baselines/stage8-verification-final/smoke/paper04-{3,4,5}uav/`. They are not renamed because their manifests preserve the executed identities. Each task contains the
MAPPO summary, DMPC manifest/source snapshot/result and `controller-comparison.json`. Research
follow-up requires the full-budget MAPPO policies and full 20-episode DMPC evaluation under the
research profile, followed by trajectory review and an analysis whose sampling units remain
explicit.
