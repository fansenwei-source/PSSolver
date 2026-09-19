# ADR 0001: strangler migration over v0.1.2

Status: accepted.

## Context

PSSolver v0.1.2 has a qualified, high-performance Plane production path.  The
separated architecture developed in Stages A--O has clearer responsibilities
and is a qualified numerical oracle, but its R320 runtime and memory footprint
failed production replacement gates.  Replacing the legacy path wholesale
would discard the strongest performance evidence in the project.

## Decision

Architecture migration uses a strangler pattern:

- the annotated v0.1.2 release is the frozen code and numerical baseline;
- `legacy_production` remains the Plane default and rollback oracle;
- the separated canary remains an opt-in numerical oracle and laboratory;
- new declarations and plans are lowered onto the existing optimized kernels
  and grouped storage;
- migration proceeds one seam at a time with compatibility facades;
- no third independent Plane runtime is created;
- candidate qualification does not authorize a default change.

## Consequences

The code temporarily contains legacy and target abstractions, but their roles
are explicit.  Early work prioritizes boundaries and adapters rather than a
visually complete rewrite.  A slower implementation cannot be promoted merely
because it has cleaner object boundaries.

## Rejected alternatives

- Replacing production with the existing separated canary: rejected by Stage
  O performance and memory evidence.
- Continuing to add features directly to legacy monoliths: rejected because
  it compounds model/geometry/backend coupling.
- Building a third runtime from scratch: rejected because it duplicates risk
  and creates another comparison target.

## Review trigger

This decision may be revisited only after an opt-in compiled runtime meets the
full Plane numerical, restart, output, memory, and H100 performance gates.
