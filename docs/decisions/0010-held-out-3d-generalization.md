# ADR 0010: Evaluate 3D pose generalization on disjoint target-pose ranges

## Status

Accepted.

## Context

A controller can appear to support arbitrary 3D formations while always starting in the target's
orientation and scale. It then learns mostly a translation. Random train/test seeds are also not a
generalization test when both sample the same pose distribution. Agent-to-target assignment and the
coordinate frame shown to a policy can materially change task difficulty and must not remain hidden
implementation details.

## Decision

- Keep initial formation geometry fixed and sample the target pose independently per episode.
- Represent pose by target-center offset, roll/pitch/yaw offset, and positive uniform scale.
- Require training and held-out pose boxes to be disjoint in at least one of seven dimensions.
- Derive pose randomness from a dedicated stream, independently of initial-position noise.
- Support stable fixed assignment and deterministic minimum-total-distance assignment.
- Support world-frame and target-aligned actor observations, critic state, and velocity actions.
- Keep simulator stepping, reward, termination, and physical metrics in the world frame.
- Put the 2x2 assignment/frame ablation on cube and use `minimum-target` for the other shapes.
- Evaluate every seed on fresh in-distribution and held-out episodes, then aggregate policy means.

## Consequences

The target-aligned representation removes nuisance orientation from what the actor must learn, while
scale remains visible through distances. Rotating a per-axis bounded action can leave the world-frame
action cube, so commands are clipped and the clip fraction is reported. Minimum-distance assignment
uses full target knowledge and is an oracle design choice, not a decentralized capability. Euler
boxes are auditable but are not uniform samples over SO(3). A smoke run validates plumbing only; it
cannot close the held-out performance gate.
