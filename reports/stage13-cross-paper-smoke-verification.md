# Stage 13 cross-paper smoke verification

The Stage 13 report builder was exercised on the three existing guarded smoke comparisons for the
3-, 4-, and 5-UAV Paper 04 tasks. This run reused saved artifacts and performed no training.

The generated bundle contained six controlled rows, five native-system rows, ten checksum-verified
evidence sources, a machine-readable report, a reviewable Markdown report, a manifest, and a project
source snapshot. Every expected controlled task was present and every controller pair retained its
task-specific comparison fingerprint.

The readiness gate correctly returned `false` because all three inputs had profile `smoke`. Their
256-transition training budgets, two evaluation episodes, zero smoke-horizon success values, and
associated standard deviations are plumbing evidence only. They must not be used to rank MAPPO and
DMPC or support a scientific claim.

Command used:

```bash
uv run uav-swarm-control build-cross-paper-report \
  --config configs/comparison/stage13_cross_paper.yaml \
  --comparison artifacts/baselines/stage8-verification-final/smoke/paper04-3uav/controller-comparison.json \
  --comparison artifacts/baselines/stage8-verification-final/smoke/paper04-4uav/controller-comparison.json \
  --comparison artifacts/baselines/stage8-verification-final/smoke/paper04-5uav/controller-comparison.json \
  --output artifacts/comparison/stage13-smoke-20260907 \
  --project-root . \
  --evidence-root ..
```

The generated artifact directory is ignored by Git. Re-run the same workflow with a new output root
after full-budget Stage 7 and Stage 8 artifacts are available.

