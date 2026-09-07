# ADR 0008: Make the corrected Paper 04 baseline explicit

- Status: accepted
- Date: 2026-09-06

## Context

Paper 04 and the supplied my-mappo tree differ in model, reward activation, geometry normalization,
and execution framework. Silent copying would make later improvements difficult to attribute.
Final-step metrics also miss collisions and excessive movement earlier in an episode.

## Decision

Compose the validated MAPPO and PyBullet schemas and require five distinct training seeds.
Implement paper-style progress reward and activate normalized formation cost. Retain the verified
feed-forward actor/critic as a named MA-PPO baseline, with every known deviation documented in
mappo-baseline.md. Keep the existing kinematic and PyBullet reward paths unchanged.

Evaluate reloaded actors on paired held-out episodes and collect metrics over the entire trajectory.
Aggregate episodes within seeds before estimating variation between seeds. Keep a scripted
proportional reference on the same episodes.

Archive source, lockfile, configuration, installed simulator identity, checkpoint hashes and training
diagnostics. Refuse accidental overwrites. Support skipping validated completed seeds without
claiming optimizer/RNG resumption. Separate smoke outputs from research outputs.

## Consequences

Later controller comparisons can share tasks, action meaning, initial states and metrics.
The full research suite is expensive and belongs on the lab machine. Passing short tests validates
engineering behavior, not policy convergence or the paper's reported advantage. LSTM, PopArt,
permutation-invariant encoding and dynamic obstacles remain explicit subsequent work.
