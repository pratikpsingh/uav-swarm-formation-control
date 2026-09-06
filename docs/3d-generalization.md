# Held-out 3D formation generalization

Stage 9 asks a sharper question than “can the code rotate a template?”:

> After training on one range of target translations, orientations, and scales, can the same local
> actor control fresh episodes in a deliberately disjoint pose range?

The implemented method is `stage9-3d-generalized-feedforward-mappo`. It retains the corrected Stage 7
feed-forward MAPPO reward, Crazyflie physics, local actors, and centralized training critic. It is a
controlled extension, not a claim that Paper 04 studied arbitrary 3D target poses.

## Geometry and episode construction

Let centered template points be rows of `P`, target center be `c`, uniform scale be `s > 0`, and the
proper Euler rotation be `R = Rz(yaw) Ry(pitch) Rx(roll)`. World targets are

```text
T = s P R^T + c
```

Initial UAV positions use the nominal unrotated/unscaled template around a separate initial center,
plus seeded position noise. They do not inherit the sampled target pose. This prevents a visually 3D
episode from reducing to the same rigid translation.

Pose sampling has its own deterministic random stream. YAML declares closed bounds for three center
offsets, three Euler offsets, and scale. Training and held-out boxes must be disjoint. Sampling is
uniform in Euler coordinates, not uniform over SO(3); results apply only to the declared ranges.

## Agent-to-target assignment

The environment preserves stable agent rows, then chooses one target per agent:

- `fixed`: agent `i` receives template target `i`;
- `minimum-distance`: the Hungarian algorithm minimizes nominal initial-to-target distance in
  `O(N^3)` time.

Minimum-distance assignment can shorten transients and reduce crossings, but it uses all origins and
targets. It is centralized oracle preprocessing. Deployment needs an equivalent coordinator or a
policy that is invariant to target permutations.

## World and target coordinate frames

The world variant retains Stage 7 inputs. The target variant changes vector coordinates:

```text
v_target = v_world R
v_world  = v_target R^T
```

The actor receives target displacement, own velocity, relative neighbor position/velocity, and masks
in target coordinates. The training-only critic receives centered/rotated joint positions,
velocities, and targets. Actor velocities are rotated back before world-frame physics.

Target framing builds rotational equivariance into the representation. It does not normalize scale;
physical distances still reveal target size. Because a rotated per-axis action cube can exceed the
world component bounds, actions are clipped and `action_frame_clip_fraction` is reported.

## Experimental design

| Configuration | UAVs | Shape | Variants |
| --- | ---: | --- | --- |
| `stage9_plane_4uav.yaml` | 4 | plane | minimum-distance + target frame |
| `stage9_pyramid_5uav.yaml` | 5 | pyramid | minimum-distance + target frame |
| `stage9_cube_8uav.yaml` | 8 | cube | fixed/minimum-distance x world/target |
| `stage9_sphere_8uav.yaml` | 8 | sphere | minimum-distance + target frame |

Cube carries the factorial ablation so assignment and frame effects can be isolated. Every variant
uses training roots 11, 22, 33, 44, 55 and a separate evaluation root. Each policy sees fresh
in-distribution and held-out evaluation episodes. For each physical metric:

```text
generalization gap = held-out mean - in-distribution mean
```

Direction depends on the metric: positive success is favorable; positive error or collision is not.
Absolute values and gaps must be read together. Pose and assignment metadata accompany every episode.

## Local smoke verification

```bash
uv run uav-swarm-control run-generalization \
  --config configs/experiment/stage9_plane_4uav.yaml \
  --config configs/experiment/stage9_pyramid_5uav.yaml \
  --config configs/experiment/stage9_cube_8uav.yaml \
  --config configs/experiment/stage9_sphere_8uav.yaml \
  --smoke --output artifacts/generalization/check --project-root .
```

Smoke retains all variants, both splits, and five seeds. It changes the budget to 256 training
transitions, 128-step rollouts, one environment, one PPO epoch, two episodes per split, CPU, and a
48-step horizon. It validates configuration, physics, learning, checkpoints, evaluation, aggregation,
provenance, and resume behavior. It cannot establish convergence.

Use `--resume` only with identical source/runtime provenance and configuration. Completed seeds need
matching checkpoint hashes. An interrupted seed does not preserve optimizer, rollout, or RNG state;
restart it in a new output root.

## Lab protocol and budget

Remove `--smoke` and use a new output root. Seven variants times five seeds times 10 million steps is
**350 million environment transitions**, or **2.45 billion agent samples** across the configured swarm
sizes. PyBullet is primarily CPU work; the GPU accelerates network updates but not the sequential
physics loop. Benchmark one seed/variant first, then schedule the frozen suite.

Do not repeatedly tune against the held-out range. If it influences changes, call it validation data
and reserve a new untouched test range for a later unbiased claim.

## Artifacts and interpretation

Each experiment writes a manifest, source snapshot, variant/seed directories, checkpoints, update
logs, raw split episodes, per-seed summaries, and across-seed statistics. Generated files belong under
ignored `artifacts/`; only small reviewed reports belong in Git.

The software gate closes when math, transforms, schemas, static checks, and every physical smoke path
pass. The scientific gate remains open until full policies are trained, curves and trajectories are
reviewed, and held-out performance is reported across independent training seeds. Zero smoke success
is not a research result, and successful smoke episodes would not prove generalization.

Recorded limitations: no dynamic obstacles, no neighbor-count ablation, all-neighbor observations,
no recurrence or permutation-invariant neighbor encoder, Euler-box sampling, centralized assignment,
per-axis velocity clipping, and no hardware deployment.
