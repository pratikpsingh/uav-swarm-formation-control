# Oracle dynamic-obstacle protocol

The dynamic-obstacle study isolates one question: does a staged obstacle curriculum improve collision-free formation
flight compared with training on no obstacles or static obstacles? It fixes the swarm to four UAVs,
the plane formation, all-neighbor sensing, the selected target-frame assignment variant, physics,
reward, optimizer, budget, and seed protocol. Neighbor count and model compression do not vary here.

## Obstacle model

Obstacles are kinematic spheres described by center `o`, constant velocity `u`, and radius `r`:

```text
o(t + dt) = o(t) + u dt
clearance(i, j) = ||p_i - o_j|| - vehicle_radius - r_j
```

A negative clearance is a sampled collision. Obstacles are intentionally not PyBullet rigid bodies:
the first experiment studies high-level avoidance with exact obstacle state, independently of contact
response, rendering, perception, and sensor error. The flight vehicles still use the pinned
Crazyflie rigid-body backend. Collision is evaluated at the 30 Hz control samples, so continuous-time
contact between samples is not claimed.

Each field is sampled near the path between initial and target centroids. A rejection check protects
the initial formation, target points, ground plane, and other obstacle spheres. Static obstacles have
zero velocity. Dynamic obstacles begin laterally displaced and cross the centroid route. Obstacle
randomness has its own derived seed stream and cannot perturb initial-state or target-pose sampling.

## Information available to the policy

Each actor gets up to `max_obstacles` nearest oracle records:

```text
[relative position (3), relative velocity (3), radius (1)] + validity mask
```

Unused slots contain exact zeros and a false mask. Every scenario—including no obstacles—therefore
has one fixed input shape and one comparable network architecture. Relative vectors use the selected
target frame; physical states and external measurements remain in world coordinates. The
centralized training critic receives padded global obstacle position, velocity, radius, and mask.
The critic is still absent during decentralized evaluation.

## Reward and termination

The corrected Paper 04 progress/formation/UAV-collision reward is retained. Per agent, the obstacle study adds:

```text
q_i = clip((safety_margin - nearest_clearance_i) / safety_margin, 0, 1)
r_i = r_base_i - proximity_weight q_i^2 - collision_penalty collision_i
```

The smooth proximity term supplies learning signal before contact. Sphere overlap terminates the
episode as failure. Reward terms are separately reported; evaluation conclusions use physical task
metrics rather than reward totals.

## Controlled training and evaluation

The three equal-budget training regimens are:

- no-obstacle control: no obstacles for the full run;
- static control: static obstacles for the full run;
- staged curriculum: no obstacle through 10%, static through 35%, slow dynamic through 65%, then
  mixed dynamic through 100%.

Scenario selection happens at reset, so an episode finishes under one coherent field and a threshold
can be crossed by at most one episode length. Each of five independent training seeds produces one
policy. Every policy is evaluated with the same episode seeds on no-obstacle, static, slow-dynamic,
and mixed-dynamic scenarios. The aggregation order is episodes within policy, then the five policy
means. The final artifact also reports seed-paired differences between regimens.

Primary outcomes are collision-free success, minimum obstacle surface clearance, mean normalized
formation error, and capped time to goal. The cap is the episode horizon for failures, preventing
successful-only timing from hiding failures. UAV-UAV collision, path, control, return, and rigid-body
metrics remain available as diagnostics.

## Commands

Validate the complete design with a bounded run:

```bash
uv run uav-swarm-control run-obstacle-study \
  --config configs/experiment/obstacle-avoidance/plane-4-uav.yaml \
  --smoke \
  --output artifacts/obstacles/check \
  --project-root .
```

Run the frozen research budget on the lab machine by omitting `--smoke`:

```bash
uv run uav-swarm-control run-obstacle-study \
  --config configs/experiment/obstacle-avoidance/plane-4-uav.yaml \
  --output artifacts/obstacles/research \
  --project-root .
```

`--resume` skips only complete seed directories whose manifest fingerprint and checkpoint digest
match. It does not restore optimizer state for an interrupted seed.

## Artifact hierarchy

```text
<output>/<profile>/<experiment>/
├── manifest.json
├── source.zip
├── summary.json
└── <regimen>/
    ├── summary.json
    └── seed-<seed>/
        ├── run.json
        ├── updates.jsonl
        ├── latest.pt
        ├── model.pt
        └── result.json
```

The manifest freezes source/runtime provenance and declares the oracle and sampled-contact
semantics. Raw episode context includes target pose, scenario index, obstacle counts, and curriculum
progress. `summary.json` contains cross-seed scenario results and paired differences.

## Current limitations

- Obstacle states are exact; there is no occlusion, delay, noise, tracking, or finite field of view.
- Spheres have prescribed constant velocities and no physical response or workspace boundary.
- Sampled collision checks can miss contact between control steps.
- The task studies one four-UAV plane; it does not establish transfer to other formations or sizes.
- Curriculum thresholds are specified, not optimized or adaptively advanced by competence.
- Smoke completion proves the software workflow, not avoidance learning or superiority.

Perception/noise, continuous collision methods, physical obstacle bodies, shape transfer, and adaptive
curricula are separate experiments so their effects can be identified.
