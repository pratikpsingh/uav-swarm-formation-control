# Architecture

## Design goal

The project will compare classical and learned swarm controllers without coupling task definitions
to one simulator or algorithm. The intended dependency direction is:

```text
configuration
     |
     v
task and formation definitions
     |
     v
environment interface <--- simulator adapter
     |
     v
controller interface <--- DMPC or decentralized policy
     |
     v
rollout and evaluation
     |
     v
metrics and experiment artifacts
```

Domain mathematics should not import PyBullet, Isaac Sim, or an RL framework. Simulator adapters
may depend on domain interfaces, but domain code must not depend on adapters. Training code may use
a centralized state, while the deployable actor must receive only explicitly declared local
observations.

## Planned package boundaries

- `algorithms`: training logic such as PPO and MAPPO.
- `controllers`: shared controller protocol and classical controller adapters.
- `environments`: simulator-independent API and simulator adapters.
- `evaluation`: reward-independent metrics and evaluation orchestration.
- `formations`: formation geometry, transforms, assignments, and errors.
- `models`: neural-network architectures used by learned controllers.
- `observations`: local observations, centralized state, masks, and normalization.
- `rewards`: individually testable reward components and composition.

Stage 1 implements simulator-independent formation geometry. Stage 2 implements typed multi-agent
observations, actions, lifecycle results, deterministic seeding, and validated experiment
configuration. Concrete environment dynamics and reward behavior remain Stage 3 work.
