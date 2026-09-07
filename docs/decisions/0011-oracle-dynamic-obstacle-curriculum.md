# ADR 0011: Isolate dynamic avoidance with oracle kinematic spheres and matched controls

## Status

Accepted.

## Context

Adding obstacle perception, contact physics, new formations, neighbor limits, and a new training
schedule simultaneously would make failure causes inseparable. A dynamic-obstacle claim also needs
controls: performance after curriculum training is uninterpretable without equal-budget policies
trained on no obstacles and static obstacles.

## Decision

- First model obstacles as seeded kinematic spheres with exact center, velocity, and radius.
- Give decentralized actors relative position, relative velocity, radius, and a validity mask.
- Pad actor and centralized-critic inputs to one configured maximum in every scenario.
- Train equal-budget no-obstacle, static-obstacle, and staged-curriculum regimens.
- Progress the curriculum from none to static, slow dynamic, and mixed dynamic at episode resets.
- Fix the four-UAV plane, all-neighbor sensing, selected target-frame variant, and all other protocol
  choices while obstacle training distribution varies.
- Evaluate every policy on all four scenarios using matched episode seeds and paired seed differences.
- Treat sampled sphere overlap as terminal failure and report reward-independent safety/task metrics.

## Consequences

The experiment can attribute differences to training obstacle distribution under the declared model.
Padding keeps architecture capacity constant, while oracle state establishes an upper-bound control
problem before perception is introduced. Kinematic obstacles are not PyBullet bodies, collision is
sampled at control frequency, and constant-velocity trajectories do not model reactions. Therefore
this stage cannot claim real sensor robustness or physical contact realism. Those are later controlled
extensions, not hidden assumptions.
