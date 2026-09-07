# Neighbor and communication study

## Question

How do requested neighbor count, achieved communication topology, sensing range, obstacle
condition, formation geometry, and swarm size affect decentralized formation control?

The neighbor and communication experiment is a controlled study, not a search for one universal neighbor count. A requested count
is only an upper bound: range limits may make fewer agents observable. Results therefore report
both the configured request and the graph that actually occurred.

## Actor representation

The actor receives padded neighbor rows containing relative position and velocity. Every valid row
passes through the same neural network. Invalid rows are removed by the Boolean mask, and the
remaining embeddings are averaged:

```text
h_i = mean({phi(relative_position_ij, relative_velocity_ij) | j is visible to i})
action_i = policy(ego_i, h_i, obstacle_observations_i)
```

If no neighbor is visible, the pooled vector is zero. Shared encoding plus a commutative mean makes
the result invariant to neighbor-slot order. Mean pooling also avoids increasing representation
magnitude merely because more rows are present. Observation capacity remains N-1 for every policy;
changing the active count does not silently change model size.

The centralized critic still receives global state during training. Only the local actor is used at
evaluation and deployment.

## Frozen research matrix

Two files vary formation and swarm size:

- `neighbor-study/plane-4-uav.yaml`: planar four-UAV formation;
- `neighbor-study/pyramid-5-uav.yaml`: spatial five-UAV formation.

Each file crosses:

- requested neighbors: 0, 1, 2, and every peer;
- sensing: 1.0 m or unlimited;
- obstacles: none or mixed dynamic.

That produces 16 evaluation conditions. Three equal-budget policies are trained per seed:

- fixed two-neighbor, unlimited-range, mixed-obstacle policy;
- fixed all-neighbor, unlimited-range, mixed-obstacle policy;
- variable-topology policy sampled uniformly across all 16 conditions at episode reset.

Five independent training seeds and 20 shared evaluation episodes per condition yield 15 policies
per task and 30 policies overall. The primary sampling unit is the independently trained policy.
Reports first average episodes within a policy and then calculate mean, sample standard deviation,
and a two-sided 95% Student-t interval across five policy means. Paired differences match training
seeds before subtraction.

## Graph definitions

An edge i -> j means actor i observes agent j. Each actor selects its nearest eligible agents, so
the graph can be directed. Connectivity, algebraic connectivity, and rigidity use the undirected
support A OR A-transpose; reciprocity reports how much of the directed graph is bidirectional.

Reported topology metrics include:

- mean, minimum, and maximum actual out-degree;
- connected-step fraction and maximum component count;
- mean and minimum Laplacian algebraic connectivity (lambda-2);
- rigidity-matrix rank, maximum rank, rank fraction, and rigid-step fraction;
- total and per-agent-step received payload bytes.

Rigidity is evaluated in the positions' numerically detected intrinsic dimension. For dimension d,
the maximum generic rank is dN-d(d+1)/2. This makes a planar formation interpretable as planar
rather than declaring it non-rigid solely because coordinates are stored in 3D.

The byte model counts six float32 neighbor values, or 24 bytes per directed edge per control step.
It excludes identifiers, timestamps, radio headers, retransmission, localization, and obstacle
sensing. It is therefore a declared payload estimate, not measured network traffic.

## Paper 02 comparison boundary

Paper 02 fixed 32 robots and compared sensed counts 1, 2, 6, 16, and 31. It reported that two was
most stable for its task. We treat that as a hypothesis to retest. Comparable concepts are success,
collision occurrence, terminal goal error, and the direction of the neighbor-count trend.

Absolute values and reward totals are not directly comparable: Paper 02 uses point-to-point
navigation, static clutter, rotor-thrust actions, and an attention encoder, whereas this project
uses formation tracking, mixed dynamic obstacles, velocity commands passed to a PID controller,
and masked-mean aggregation. These differences are embedded in each run manifest.

## Commands

Bounded integration run:

```bash
uv run uav-swarm-control run-communication-study \
  --config configs/experiment/neighbor-study/plane-4-uav.yaml \
  --config configs/experiment/neighbor-study/pyramid-5-uav.yaml \
  --smoke \
  --output artifacts/communication/check \
  --project-root .
```

Run the research budget on the lab machine by omitting `--smoke`. Use `--resume` only to skip
complete seed directories with matching provenance and checkpoint digests; it does not reconstruct
an interrupted optimizer.

## Interpretation

First verify learning and safety metrics. Then compare performance against actual degree, connected
fraction, lambda-2, rigidity, and bytes. A correlation does not prove that a graph property caused
performance, but it can identify plausible mechanisms and motivate a narrower intervention. Do not
select a neighbor count from reward alone or from a single seed.

Smoke runs establish software integration only. They cannot establish an optimal degree, robust
dynamic-obstacle avoidance, or superiority over Paper 02.
