# Cross-paper comparison protocol

Stage 13 separates two questions that cannot be answered by one leaderboard.

The **controlled common-environment track** asks how implemented controllers behave when the task,
simulator, initial-condition protocol, horizon, and metrics are identical. It currently accepts the
guarded Stage 8 comparison artifacts for the corrected Paper 04 feed-forward MAPPO baseline and the
clean-room DMPC-Swarm adaptation. Each artifact already proves that the two controllers share a
comparison fingerprint.

The **native-system context track** asks what each publication or supplied native repository actually
did. It covers Paper 01 and Paper 02 in QuadSwarm, Paper 03 in OmniDrones/Isaac Sim, Paper 04 in a
custom Gym/PyBullet/Ray/RLlib system, and the native DMPC-Swarm software-in-the-loop stack. Native
numbers retain their own metric definitions and are never ranked against the common-environment
numbers.

## Why the tracks must remain separate

A policy that emits motor thrust solves a different control problem from one that emits desired body
rates or desired velocity. A local actor with two nearest neighbors has a different information budget
from an optimizer that receives all positions, velocities, assigned targets, and identities. Likewise,
one billion direct-thrust simulator transitions cannot be treated as equivalent to 10 million
velocity-command transitions in another physics engine.

Even identical metric names can hide different random variables. Paper 02's success is an agent-level
native quantity. Paper 03's collision-free rate combines arrival and collision conditions. This
project's `collision_free_success` is an episode-level common-harness metric. Stage 13 therefore
compares values only when their protocol fingerprint and metric implementation are shared.

## Evidence catalog

The checked-in `configs/comparison/stage13_cross_paper.yaml` catalog requires every native row to
state:

- environment and simulator version or an explicit `not-reported` status;
- action semantics and observation access;
- training budget, including `not-applicable` for classical optimization;
- seed evidence and uncertainty evidence;
- centralized or decentralized execution;
- evaluation protocol, selected native result, source locator, and comparability boundary.

Evidence has one of four statuses: `reported` means the publication states it, `measured` means it was
audited from the supplied repository, `not-reported` preserves a genuine documentation gap, and
`not-applicable` marks a concept that does not apply. The report command verifies SHA-256 checksums of
every source file before using the catalog. A changed paper or repository source therefore requires a
new audit and an intentional config update.

## Controlled artifact checks

For every supplied `controller-comparison.json`, the report builder follows and validates the MAPPO
summary, DMPC result, and MAPPO manifest. It rejects:

- mismatched protocol fingerprints;
- unknown method identities;
- MAPPO/DMPC profile disagreement;
- missing configured metrics;
- duplicate or unexpected tasks;
- missing referenced artifacts.

The report repeats the environment, simulator revision, actions, observations, budgets, seeds,
uncertainty unit, and execution architecture on every result row. MAPPO sample standard deviation is
across independent trained-policy seed means. DMPC sample standard deviation is across deterministic
evaluation episodes. They are intentionally not interpreted as equivalent estimators.

## Build the current smoke report

```bash
uv run uav-swarm-control build-cross-paper-report \
  --config configs/comparison/stage13_cross_paper.yaml \
  --comparison artifacts/baselines/<run>/smoke/paper04-3uav/controller-comparison.json \
  --comparison artifacts/baselines/<run>/smoke/paper04-4uav/controller-comparison.json \
  --comparison artifacts/baselines/<run>/smoke/paper04-5uav/controller-comparison.json \
  --output artifacts/comparison/<report-run> \
  --project-root . \
  --evidence-root ..
```

The output directory is immutable: an existing report is never overwritten. It contains `report.md`
for review, `report.json` for analysis, `manifest.json` for input/config/source provenance, and
`source.zip` for the project code snapshot. The bundle is research-ready only when every expected
task is present and all controlled inputs use the `research` profile. Complete smoke coverage remains
`research_ready: false` by construction.

## What remains for the thesis

Stage 13 completes the software and evidence protocol, not the expensive empirical study. Run the
full Stage 7 and Stage 8 lab protocols first, inspect convergence and failures, build a research-profile
report, and freeze it before introducing Stage 9-12 treatments. Paper 01-03 require explicit adapters
to this project's common action/observation/environment contract before they can enter the controlled
track. Until those adapters exist, their native evidence belongs only in the native-system track.

