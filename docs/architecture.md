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
- `obstacles`: simulator-independent obstacle state, sampling, motion, and sphere clearance.
- `observations`: local observations, centralized state, masks, and normalization.
- `rewards`: individually testable reward components and composition.

Stage 1 implements simulator-independent formation geometry. Stage 2 implements typed multi-agent
observations, actions, lifecycle results, deterministic seeding, and validated experiment
configuration. Stage 3 implements the first-order kinematic environment, structured observations,
decomposed rewards, independent metrics, scripted controller, and episode runner. Stage 4 validates
the project-owned PPO implementation on a known single-agent task. Stage 5 applies that foundation
to parameter-shared MAPPO with a local actor and centralized critic. Stage 6 adds a typed rigid-body
backend, a pinned gym-pybullet-drones bridge, and a PyBullet environment that reuses the same
observation, reward, termination, and metric builders as the kinematic environment.

The corrected Paper 04 baseline composes both configuration schemas, replaces absolute-distance
shaping with progress reward in its own environment, and evaluates complete trajectories through a
controller protocol. Training has an optional update callback for progress records and diagnostic
checkpoints. The experiment runner owns source/runtime provenance, completed-seed validation, and
aggregation across independent training seeds. Scientific validation is separate from smoke tests.

Stage 8 adds optional `StateAwareController` and `EpisodeResettableController` capabilities without
weakening the original local-observation `Controller` protocol. The common evaluator performs
structural dispatch: learned actors continue to receive only local observations, while model-based
controllers can explicitly request centralized state. This information difference is recorded in
the result rather than hidden in the environment.

The clean-room DMPC adapter keeps trajectory mathematics in `controllers`, strict controller
choices in `configuration`, and experiment orchestration in `evaluation`. Both MAPPO and DMPC use a
shared provenance module and a hashed comparison protocol. The hash covers environment physics,
evaluation schedule, horizon and metric schema; result comparison fails closed when hashes differ.
Controller-specific solver diagnostics remain outside common task metrics.

Stage 9 keeps pose sampling and linear assignment in the simulator-independent `formations`
package. A specialized physical environment samples a target pose from its own deterministic random
stream, leaves the initial formation geometry independent, assigns target points, and optionally
expresses actor/critic inputs in the target frame. World-frame physics, rewards, terminal conditions,
and external metrics remain unchanged. The experiment runner trains independent seeds, evaluates
fresh episodes from in-distribution and disjoint held-out pose ranges, and reports held-out-minus-
in-distribution gaps without treating smoke policies as scientific evidence.

Stage 10 keeps kinematic sphere mathematics in `obstacles`, curriculum/protocol validity in
`configuration`, oracle observation/reward/termination composition in `environments`, and multi-seed
paired comparisons in `evaluation`. Obstacle randomness has an independent stream. Fixed-width masks
keep actor and critic architectures identical across no/static/dynamic cases. The Crazyflie dynamics
remain physical, while obstacle bodies and perception are explicitly outside this first experiment.

Stage 11 separates padded observation capacity from active communication topology. The actor's
shared neighbor encoder and masked mean live in models; pure adjacency, connectivity, spectral,
rigidity, and payload calculations live in communication; per-episode selection lives in the
specialized environment; and fixed-versus-variable multi-seed inference lives in evaluation.
Versioned checkpoints persist the structured encoder without breaking earlier flat actors.
