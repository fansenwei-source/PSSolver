# ADR 0006: identity, provenance, and qualification

Status: accepted.

## Context

Scientific parameters, discretization choices, implementation selectors, and
run controls currently coexist in one large run specification and metadata
document.  Performance work can therefore look like a model change, while
duplicated application metadata can drift from the implementation.

## Decision

Provenance separates four identities:

- scientific: equations, parameters, conventions, and physical BCs;
- discretization: grid, basis, dtype, storage semantics, dealiasing,
  zero-mode policy, and time integrator;
- execution: backend, concrete transform/solver implementations, compile
  policy, device policy, and workspace strategy;
- run: initial-condition realization, seed, duration, save schedule, restart,
  and output location.

Each responsible layer emits structured metadata.  Applications compose these
records and do not restate implementation facts manually.

Qualification is proportional to the change:

- mechanical extraction: byte-identical numerical outputs and unchanged
  operation order;
- intentional equivalent reordering: prospectively frozen relative-L2 and
  Linf tolerances;
- new operator or BC: manufactured solution, boundary residual, and
  convergence tests;
- hot-path change: balanced GPU A/B, memory, call counts, fallback, and finite
  output gates;
- default change: a separate promotion commit, implicit-default smoke, and
  explicit review;
- model promotion: benchmark-scale stability and statistical validation.

## Checkpoint policy

Checkpoints store semantic evolved state, clock, integrator history, identity
records, and explicitly persistent algebraic state.  They do not promise a
permanent internal packing index, compiled graph, transform cache, or transient
workspace.  Incompatible schemas fail explicitly unless a versioned adapter
exists.

## Consequences

An execution optimization can be audited without changing scientific
identity.  Qualification evidence cannot silently authorize a new default,
and tolerances cannot be relaxed after observing a result.

