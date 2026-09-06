# Policy compression and deployment protocol

## Question

How small can the decentralized Stage 11 actor become without losing collision-free formation
performance? This is a constrained optimization problem, not a file-compression contest. A tiny actor
that crashes or fails to reach the goal is not a successful deployment.

## Teacher gate

The research command accepts only a Stage 11 checkpoint whose SHA-256 digest matches its result
file, whose seed and manifest fingerprint match checkpoint metadata, whose resolved task
configuration equals the requested task, and whose predeclared condition metrics pass the YAML
gates. Smoke mode still checks identity and provenance, but labels every output
`plumbing-only-smoke-teacher`; it cannot produce a scientific result.

Select the teacher after the full Stage 11 runs, using the predeclared safety and task gates rather than
choosing whichever checkpoint happens to compress best. This prevents post-hoc selection bias.

## Controlled comparisons

All students retain the padded masked-mean neighbor interface so permutation invariance and local
information boundaries do not change. The matrix compares three feed-forward widths, a GRU, an
LSTM, a 50% structured-channel mask followed by fine-tuning, and TorchAO dynamic INT8. Every
student learns the deterministic teacher's actions. Complete episodes—not individual rows—are split
into training and validation sets, stratified by condition when each condition has multiple episodes, and recurrent state starts at zero for each episode.

Five independent distillation seeds quantify optimization variation. Each student is then run in closed
loop on the same Stage 11 conditions and episode seeds as the teacher. Imitation MSE is diagnostic;
the decision metrics are collision-free success, collisions, formation error, clearance, path length,
smoothness, and time to goal. Closed-loop evaluation is essential because small action errors can
change later observations and accumulate.

Structured masking zeroes whole hidden channels and keeps them zero during fine-tuning. The current
dense graph may not shrink, which is reported rather than hidden. A backend-specific channel-removal
pass is a future optimization. INT8 uses TorchAO's dynamic per-token activation and per-channel weight
configuration. Quantization is a separate treatment, not silently applied to every candidate.

## Systems measurements

Each actor-only artifact reports parameters, logical parameter bytes, serialized graph bytes, SHA-256,
and export equivalence. A fresh process measures single-agent host CPU latency and peak process RSS.
These values are useful for screening candidates but are explicitly not claimed as embedded results.
Energy is recorded only when supplied by an external target measurement; the software refuses to
invent joules from wall-clock time.

A final target study must state the board, clock, runtime/backend, compiler flags, control frequency,
power instrument, sampling procedure, warm-up, number of measurements, and whether flash/RAM
include runtime and operator code.

## Paper 02 comparison boundary

Paper 02 deployed on Crazyflie 2.1 with a 168 MHz CPU and 192 KB RAM. It reduced encoder and
attention hidden sizes to 10, used single-head attention, and reported a 1,820-parameter, 7 KB model
running in 0.35 ms onboard. Its Table II compares training from scratch with policy distillation over
20 episodes for eight robots at 20% obstacle density. Our task, observation encoder, dynamics, and
artifact format differ, so these numbers are context—not a direct leaderboard. Stage 12 mirrors the
important discipline: report task quality and systems cost together.

## Commands

Install the optional supported quantization path:

```bash
uv sync --extra deployment
```

Exercise every candidate and all five seeds with bounded budgets, using a matching Stage 11 smoke
teacher:

```bash
uv run --extra deployment uav-swarm-control run-deployment-study \
  --task configs/experiment/stage11_plane_4uav.yaml \
  --deployment configs/deployment/stage12_policy_compression.yaml \
  --teacher-checkpoint artifacts/communication/<run>/smoke/<task>/<regimen>/seed-11/model.pt \
  --teacher-result artifacts/communication/<run>/smoke/<task>/<regimen>/seed-11/result.json \
  --smoke \
  --output artifacts/deployment/<run> \
  --project-root .
```

Remove `--smoke` only when both teacher files come from the matching full research task and pass the
configured gates. The command refuses to overwrite an existing study directory.

Read `manifest.json` first, `teacher/result.json` second, then each candidate `summary.json`. The root
`summary.json` is the comparison index. Do not rank candidates solely by host latency or artifact
bytes; first reject those that fail the task requirements.
