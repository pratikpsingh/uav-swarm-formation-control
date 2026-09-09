# Manuscript red-line responses and evidence status

This document answers the editorial text in
**papers/2026_Distributed_MARL_Submission.pdf** and states what the current repository can actually
support. It is a drafting aid, not a results section. Numerical performance claims must be replaced
by regenerated immutable artifacts from frozen experiments.

Status terms:

- **Implemented/software-tested:** the code path and bounded tests exist.
- **Needs research run:** the mechanism exists, but a full multi-seed experiment has not established
  the claim.
- **Not implemented:** do not claim it in the paper.

## 1. Why compare MARL with DMPC?

**Suggested manuscript answer.** DMPC is not inherently “poor.” Its main deployment cost is that each
UAV repeatedly solves a constrained finite-horizon optimization problem online and normally
exchanges current state or predicted trajectories with neighbors at every replanning instant. The
cost therefore depends on horizon length, solver convergence, constraint count, update rate, and
neighborhood size. A trained decentralized MARL actor moves most optimization offline: online
execution is a fixed-size neural-network forward pass followed by the low-level controller. If each
actor uses at most \(k\) received neighbor records, communication can also be bounded per UAV.

The trade-off must remain explicit. MARL training is expensive, learned policies do not automatically
provide MPC-style constraint guarantees, and both approaches require the same sensing/communication
assumptions to be compared fairly. The paper should report measured deadline misses, latency, bytes,
success, clearance, and formation error for both controllers on matched tasks and hardware. It
should not use the blanket statement “DMPC is poor.”

**Status:** a clean-room DMPC reference and common physical metrics exist. A matched target-hardware
latency/communication comparison still needs to run.

## 2. Do UAVs construct the initial formation using local or global knowledge?

**Suggested manuscript answer.** At execution time, each UAV receives its own state, the displacement
to its assigned active target, and bounded local neighbor/obstacle records. The shared actor does not
receive the centralized critic state. The target template and UAV-to-target assignment are currently
computed by the simulator or a pre-mission assignment service. Consequently, continuous control is
decentralized, but target allocation is not yet a fully distributed consensus algorithm.

The mission starts the UAVs in a planar near-ground arrangement at 0.08 m, assigns targets by global
minimum-total-distance matching, and uses the phases

~~~text
CONSTRUCTING -> HOLDING -> NAVIGATING -> COMPLETE
~~~

Construction is harder than starting in formation because simultaneous 3D ascent can create
intersecting paths, transient loss of rigidity/connectivity, and inter-agent collisions. A naive
straight-line controller also lacks recovery behavior when tracking errors or external disturbances
separate the agents.

**Status:** near-ground construction, assignment, holding, and waypoint navigation are
implemented/software-tested. Learning quality needs full runs. Fully distributed target assignment
is not implemented and must not be claimed.

## 3. How is the varying neighbor set defined?

For UAV \(i\), let \(p_i(t)\) be its position, \(R_c\) the sensing/communication radius, and \(k\) the
configured cap:

\[
C_i(t)=\{j\ne i:\lVert p_j(t)-p_i(t)\rVert_2\le R_c\},
\]

\[
N_i(t)=\operatorname{TopK}_{j\in C_i(t)}
       \bigl(-\lVert p_j(t)-p_i(t)\rVert_2,k\bigr).
\]

The actual observed count is \(k_i(t)=|N_i(t)|\le k\). If \(R_c\) is unlimited, the cap is the only
restriction. If fewer than \(k\) agents are in range, unused slots are zero-padded and masked. Ties
must be resolved deterministically. The selection is generally directed: \(j\in N_i\) need not imply
\(i\in N_j\).

**Status:** radius-plus-nearest-\(k\), masks, actual-degree metrics, reciprocity, connectivity,
algebraic connectivity, numerical rigidity, and payload accounting are implemented.

## 4. What is the relationship among swarm size, neighbors, and formation error?

There is no universal formula \(e=f(N,k)\) for a learned controller. Error also depends on formation
geometry, sensing radius, graph realization, policy capacity, training distribution, obstacle
condition, and dynamics. The manuscript's U-shaped neighbor curve is therefore an empirical
hypothesis, not a mathematical law.

For an all-to-all directed exchange, the number of received records per update is

\[
M_{\mathrm{all}}=N(N-1).
\]

With an unlimited-range top-\(k\) cap,

\[
M_k=Nk,\qquad
\text{saving}=1-\frac{k}{N-1}.
\]

With a \(b\)-byte record, nominal received payload is \(bM_k\). Fixed \(k\) makes payload
\(O(N)\); keeping a fixed normalized degree \(k/(N-1)\) makes it \(O(N^2)\). Range loss can only
reduce the actual payload, and the code records that actual value.

For an undirected graph in generic 3D, a necessary edge-count condition for infinitesimal rigidity is

\[
|E|\ge 3N-6,\qquad
\bar d=\frac{2|E|}{N}\ge 6-\frac{12}{N}.
\]

This condition is not sufficient. It concerns undirected geometric edges, not the requested outgoing
top-\(k\) value. Directed selections must first be symmetrized, and their rigidity rank must be
computed at the actual positions. Even satisfying the count does not ensure connectivity, good
conditioning, collision avoidance, or good learned performance.

The controlled analysis should distinguish:

1. **Inference ablation:** one variable-topology checkpoint evaluated at several \(k\).
2. **Training ablation:** an independently trained policy for each fixed \(k\).
3. **Generalization:** a variable-topology policy evaluated on unseen \((N,k)\) pairs.

A suitable preregistered model is

\[
e=\beta_0+f\!\left(\frac{k}{N-1}\right)+\beta_N\log N+
\beta_I\log N\,f\!\left(\frac{k}{N-1}\right)+
\beta_F+\beta_O+u_{\mathrm{seed}}+\epsilon,
\]

with formation \(F\), obstacle condition \(O\), and training seed as the independent replication
unit. Also plot error against actual degree, connected fraction, \(\lambda_2\), rigidity rank, and
bytes. Freeze the form of \(f\) before final evaluation.

**Status:** same-family sphere configurations now cover \(N=4,8,16,32\) and valid grids of \(k\).
They are a feed-forward set-encoder scaling baseline. The recurrent study currently uses \(N=8\).
Full training and statistical analysis have not run.

## 5. Should communication cost appear in the reward?

A normalized candidate is

\[
r_{\mathrm{comm}}(t)=-\lambda_b\,
\frac{B_t}{N(N-1)b},
\]

where \(B_t\) is measured received payload. This is meaningful only when the policy or a separate
gating mechanism can change communication. If \(k\) is fixed for the whole episode, this term is
constant and cannot teach a better policy; it only shifts returns. First establish the accuracy-byte
Pareto curve with controlled \(k\), then test learnable/event-triggered communication as a distinct
ablation.

**Status:** bytes are measured; a communication penalty is intentionally not yet part of the reward.

## 6. Static/dynamic obstacles and wind

**Suggested manuscript answer.** The first obstacle study uses exact simulator-provided spherical
position, velocity, radius, and a validity mask. It compares clear, static, slow-dynamic, and
mixed-dynamic conditions and measures obstacle surface clearance and collisions in addition to
formation error. This isolates controller behavior before perception is introduced.

Wind, sensor noise, delay, occlusion, finite field of view, stale tracks, and physical contact bodies
are separate robustness factors. They must not be described as handled by the current method.

**Status:** oracle static/dynamic obstacle training and evaluation are implemented; wind and
perception robustness are not implemented.

## 7. Why do waypoints help?

**Suggested manuscript answer.** Waypoints do not make a straight global route geometrically novel.
Their purpose is distribution control: a policy trained with 1–3 m target displacements can receive
an out-of-distribution 20 m displacement under direct navigation. One-metre receding waypoints keep
the goal-displacement component near its training scale while preserving the same final route.

Use the same checkpoint and paired episode seeds at 3, 5, 10, and 20 m. Report collision-free
success, capped completion time, formation error, path length, and action-change RMS. Waypoint
benefit is supported only if the paired results improve; the manuscript's present table must not be
retained without traceable artifacts.

**Status:** waypoint mission logic is implemented. The direct-versus-waypoint paired runner/analysis
still needs to be completed.

## 8. Why can straight-line motion still have abrupt actions?

A straight centroid path does not imply smooth individual control. Gaussian exploration, corrections
to target error, changing neighbor membership, coupled collision avoidance, numerical control
updates, and low-level PID tracking can all produce rapidly varying commands. The smoothing term

\[
r_{\mathrm{smooth},i}(t)=-\lVert a_i(t)-a_i(t-1)\rVert_2^2
\]

should be evaluated by training matched policies with and without it. Compare physical action-change
RMS, formation error, path length, collisions, and success—not the smoothing reward alone.

**Status:** the reward component and action-change metrics exist. The configured recurrent study
currently sets its weight to zero, so the manuscript cannot claim a recurrent smoothness advantage
until an equal-budget ablation is added and run.

## 9. How much has the model been reduced?

The active 8-UAV masked-set/three-obstacle recurrent actor is larger than the manuscript's 51-input
flat actor because it includes a neighbor encoder and obstacle/mask inputs. The actor-only teacher,
excluding critic and exploration log standard deviation, has:

| Actor | Parameters | Logical FP32 weights | State per UAV |
| --- | ---: | ---: | ---: |
| recurrent teacher (256 LSTM) | 547,044 | 2,188,176 B | 2,048 B |
| FF-16 student | 1,476 | 5,904 B | 0 |
| FF-32 student | 4,484 | 17,936 B | 0 |
| FF-64 student | 15,108 | 60,432 B | 0 |
| GRU-32 student | 10,628 | 42,512 B | 128 B |
| LSTM-32 student | 13,700 | 54,800 B | 256 B |

These are architecture sizes, not accepted deployment results. Structured zeroing does not reduce a
dense artifact unless the runtime has sparse storage/kernels. INT8 logical weight size does not equal
the final binary size. The pipeline therefore records parameter bytes, serialized artifact bytes,
state bytes, host latency, export equivalence, and post-export behavior separately.

**Status:** recurrent trajectory distillation, compact FF/GRU/LSTM candidates, actor-only export, and
direction-plus-speed execution are implemented/software-tested. No candidate is accepted until a
full recurrent teacher passes the declared behavior gates. Target flash, peak RAM, latency, and
energy remain unmeasured.

## 10. Can the same \(N\) UAVs change formation to pass an obstacle?

Yes, provided both formations have exactly \(N\) target points and a safe correspondence is defined.
The implemented baseline performs minimum-distance assignment between nominal and avoidance targets,
then interpolates

\[
F_{\mathrm{active}}(\alpha)=(1-\alpha)F_{\mathrm{nominal}}+
\alpha F_{\mathrm{avoid}},
\quad 0\le\alpha\le1,
\]

with a rate-limited \(\alpha\). Its state machine is

~~~text
NOMINAL -> DEFORMING -> AVOIDING -> RESTORING -> NOMINAL
~~~

The first avoidance template is a vertical column. A deterministic obstacle-clearance trigger chooses
when to morph and restore; MAPPO tracks the active targets. This is deliberately an interpretable
high-level baseline. It does not yet prove that every interpolation is collision-free or that the
policy learned discrete shape selection.

**Status:** same-\(N\) morphing/restoration and distance-minimizing correspondence are
implemented/software-tested. Full obstacle trials and failure inspection are required.

## 11. Where does neighbor information matter if waypoints lie on a straight line?

The waypoint only translates the mission reference. It does not tell one UAV where the other UAVs
actually are. Neighbor state is used to correct relative geometry, preserve spacing, avoid
inter-agent collisions, react to lagging or disturbed agents, and recover when the formation breaks.
The decisive experiment is not a picture of a straight route; it is a controlled perturbation or
neighbor dropout followed by measured recovery.

**Status:** one-step velocity perturbation and sustained-recovery metrics are implemented. Publication
trajectories/snapshots have not yet been generated.

## 12. What visual evidence should be added?

Add synchronized 3D trajectory plots and snapshots for:

- construction from the ground;
- the same episode immediately before, during, and after a controlled perturbation;
- clear/static/dynamic obstacle avoidance;
- nominal-to-column-to-nominal morphing;
- low-\(k\) versus all-neighbor execution;
- direct versus waypoint navigation.

Each figure must identify checkpoint hash, training seed, evaluation seed, condition, and units.
Plot neighbor graph edges only when they correspond to the actual recorded selection. Do not
manually recreate trajectories.

**Status:** episode metrics and contextual formation identifiers are persisted; artifact-only paper
plot generation remains to be implemented.

## 13. Real-device statement

Replace “the INT8 payload fits comfortably” with a conditional statement. A weight estimate alone
does not include inference code, tensor metadata, allocator overhead, activation/scratch memory,
firmware growth, communication buffers, or timing. The deployment claim is complete only after
compiling the selected model for one confirmed processor, measuring the final binary and peak RAM,
meeting the control deadline, rerunning simulator/flight behavior, and measuring energy if energy is
claimed.

**Status:** portable host export and benchmarks exist. Crazyflie/nRF52840 deployment is not done. The
primary target is still an open decision.

## Claims that must be revised before submission

Until full immutable artifacts exist, change “results demonstrate,” “generalizes to 30–40/128
drones,” “averaged over 600 runs,” “optimal neighbors,” and “fits comfortably” to hypotheses,
planned comparisons, or implementation descriptions. The abstract, contribution list, experiment
section, tables, figures, and conclusion must all use the same evidence boundary.

## Recommended execution order

1. Run one recurrent **base** smoke and inspect its manifest, checkpoint reload, and result JSON.
2. Run **pooled-formations**, **mission**, and **morphing** smoke separately.
3. Pilot one seed/one regimen long enough to estimate wall time and verify learning.
4. Freeze thresholds, budgets, seeds, and direct/waypoint plus smoothing ablations.
5. Run the \(N\)-by-\(k\) feed-forward pilot; decide whether a recurrent multi-size study is
   affordable.
6. Launch full independent-seed campaigns only after the pilot gates pass.
7. Compress only a teacher that passes predeclared held-out gates.
8. Generate paper tables/figures solely from immutable artifacts.
