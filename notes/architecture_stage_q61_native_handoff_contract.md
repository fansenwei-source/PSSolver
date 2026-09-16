# Architecture Stage Q.6.1: native handoff contract

Stage Q.6.1 converts the completed Q.6 data-movement map into one bounded
experimental design. It remains design-only and changes no tensor runtime,
solver, equation, transform, integrator, restart rule, or production default.

## Contract

The proposed abstraction is a producer-owned
`BoundarySignatureNativeSegment`. A segment contains a contiguous component
run with one representation space and one projected-boundary signature. Its
flattened component/batch leading dimension is already a valid transform
input view, so constructing the handoff performs no copy.

The scheduler may transform multiple native segments independently and restore
the consumer-visible component order through metadata and views. It may not:

- concatenate native segments;
- copy them into a workspace;
- mix incompatible boundary signatures;
- retain tensor references after generation invalidation;
- silently fall back to `copy_cat`.

This deliberately avoids the rejected Stage Q.4 preallocated-workspace route.
The segment is canonical producer output, not a second destination populated
by copying an already-produced tensor.

## Three-source scope

The design is accepted only as a joint candidate for all three Q.6 sources:

1. molecular-field and Q-gradient native spectral segments supply the stress
   dependency island;
2. velocity-gradient native spectral segments supply the remaining physical
   explicit-RHS dependencies under the frozen execution order;
3. the compiled explicit-RHS evaluator supplies native physical output
   segments for the forward projected transform.

A single-source candidate remains closed because no individual source passed
the earlier screening threshold. The existing final explicit-RHS spectral
stack is not part of the measured `copy_cat` source and is not claimed as
eliminated.

## Primary risk

Avoiding cross-owner concatenation can divide a formerly combined transform
batch into more native segments. Copy traffic may fall while transform and
kernel launch counts rise. Therefore throughput cannot be predicted from the
3.45% measured subregions. A Q.6.2 candidate must measure balanced R320 H100
profiles and reject the design if launch fragmentation offsets the copy
reduction.

Q.6.1 authorizes only an experimental Plane-shadow Q.6.2 implementation. It
does not authorize production promotion, generic solver changes, Channel
changes, or default changes.
