# Research design

## Objective

Develop and evaluate a decentralized recurrent controller for multi-UAV formation construction,
maintenance, navigation, obstacle avoidance, topology-aware communication, shape reconfiguration,
and constrained deployment.

## Research questions

### Three-dimensional formation

Can one shared recurrent policy maintain cube, plane, sphere, and pyramid formations across unseen
target translations, orientations, scales, and selected swarm sizes?

Primary comparisons:

- pooled multi-formation policy versus per-formation policy;
- fixed versus minimum-distance target assignment;
- world-frame versus target-frame representation;
- in-distribution versus held-out target poses.

### Static and dynamic obstacle avoidance

Does training distribution or a clear-to-static-to-dynamic curriculum improve collision-free success
under matched obstacle scenarios?

The first experiments use exact kinematic sphere states. Perception noise, delay, occlusion, and
physical obstacle response require separate treatments.

### Actor compression

What is the smallest actor that preserves declared collision-free success and formation-error
requirements?

Parameter count, serialized bytes, recurrent state, peak RAM, latency, energy, and post-export task
behavior must be measured separately.

### Swarm size and communication topology

How do swarm size, requested neighbors, actual graph degree, connectivity, rigidity, and nominal
payload relate to formation error and collision-free success?

The analysis must distinguish:

- changing neighbor availability during inference;
- training independent fixed-neighbor policies;
- generalizing a variable-topology policy to unseen `(N, k)` combinations.

### Obstacle-triggered formation morphing

Can a swarm transition from a nominal shape to an avoidance shape, pass an obstacle, and restore its
original shape without UAV-UAV or obstacle collisions?

A deterministic high-level shape selector provides the initial interpretable baseline. The learned
policy controls continuous motion toward active targets.

## Evaluation principles

- freeze configurations before final evaluation;
- use multiple independent training seeds;
- use held-out paired evaluation seeds;
- report per-seed values, means, sample standard deviations, and failure cases;
- compare controller outputs using common physical metrics rather than internal reward/objective;
- keep native-paper evidence separate from common-environment experiments;
- generate publication plots and tables from immutable result artifacts.

## Core metrics

- collision-free success;
- UAV-UAV and obstacle collision rates;
- minimum clearance;
- position RMSE;
- normalized formation-shape RMSE;
- construction, recovery, deformation, and restoration time where applicable;
- path length and action-change RMS;
- actual degree, connectivity, algebraic connectivity, rigidity, and bytes;
- training transitions, agent samples, wall time, parameters, model bytes, and inference latency.

## Claim boundary

A smoke run validates software integration only. A scientific claim requires full frozen budgets,
independent-seed completion, held-out evaluation, curve and trajectory inspection, uncertainty, and
documented limitations.

