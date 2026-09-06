# ADR 0012: Masked mean aggregation for the neighbor study

## Status

Accepted.

## Context

Flattening padded neighbor rows makes the actor depend on arbitrary slot order and makes policies
with different active counts difficult to interpret. Stage 11 must vary communication conditions
without changing observation capacity or giving the actor privileged global state.

## Decision

Encode every neighbor row with one shared MLP, multiply by its validity mask, and mean-pool valid
embeddings. Concatenate the result with ego and obstacle features before the policy MLP. Keep N-1
padded slots, but select active neighbors and range per episode. Measure the resulting directed graph
independently of policy inputs.

Use mean rather than sum so embedding scale is not mechanically proportional to degree. Use a zero
pooled vector when no neighbor is available. Persist the encoder specification in version-two MAPPO
checkpoints while retaining support for version-one checkpoints.

## Consequences

Neighbor ordering no longer affects actions, one architecture accepts variable active counts, and
fixed-versus-variable training comparisons have equal model capacity. Mean pooling intentionally
cannot preserve cardinality by itself; the actor must infer consequences through observed state,
while degree is retained in evaluation metrics. Attention and richer set encoders remain future
controlled architecture comparisons.
