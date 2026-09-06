# ADR 0013: Compress only a validated decentralized actor

## Status

Accepted.

## Context

Compressing an undertrained smoke policy can produce impressive size and latency numbers that have
no research meaning. Deployment measurements are also easy to misstate: a training checkpoint
contains a centralized critic, a portable graph is not firmware, dense storage does not automatically
benefit from zero weights, and laptop RSS is not microcontroller RAM. Recurrent policies add another
risk because hidden state can leak across episodes or be omitted from the deployment interface.

## Decision

A research compression run must cryptographically bind a Stage 11 checkpoint to matching evaluation
evidence and resolved configuration, then pass predeclared behavioral gates. Smoke runs are always
non-scientific. Only the deterministic decentralized actor is exported. The actor preserves masked
permutation-invariant neighbor aggregation. Feed-forward, GRU, and LSTM students use complete-episode
distillation splits; recurrent state is caller-owned and reset per episode.

The comparison includes controlled width changes, structured whole-channel masks with fine-tuning,
and TorchAO INT8. Actor graphs use `torch.export`; target compilation is a later explicit boundary.
Reports keep logical parameters, serialized bytes, host peak RSS, host latency, closed-loop task
metrics, and externally measured energy separate.

## Consequences

The pipeline cannot manufacture deployment evidence from a convenient checkpoint, and the critic
cannot inflate actor size. Results remain auditable across seeds. The portable artifact is useful for
backend work but does not establish Crazyflie feasibility. Structured pruning may expose sparsity
without reducing dense artifact size; a target-aware compaction pass must demonstrate any claimed
flash reduction. Actual energy and embedded RAM/latency remain pending until target hardware and a
power measurement setup exist.
