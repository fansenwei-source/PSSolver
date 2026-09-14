# Architecture Stage N.4.1: compiled scheduler hot path

## Purpose

Stage N.4 established one execution-policy authority and extracted transform
and physical-island scheduling from the legacy execution context. Its bounded
H100 qualification preserved numerical results, transform counts, algebraic
field lifetime, and memory use, but the candidate/control mean timestep ratio
was `1.036939166`, above the frozen `1.03` non-regression limit.

Stage N.4.1 removes repeated Python orchestration from that scheduler without
changing its ownership or scientific semantics. It is a recovery candidate
for the existing Stage N.4 qualification, not a production promotion.

## Changes

- A `BoundarySignatureTransformScheduler` is now created once per
  `LegacyAlgebraicSolverContext` and shared by the algebraic-field and explicit
  RHS adapters.
- Each stable `(direction, ordered component names)` signature compiles one
  immutable `CompiledProjectedTransformPlan`.
- A compiled plan retains the complete compatibility keys and ordered batch
  indices. Boundary lookup, dtype/device string conversion, and grouping are
  therefore cold-path operations.
- Internal hot paths consume ordered tensor tuples directly. Public mapping
  and `ScheduledProjectedTransforms` facades remain available for callers that
  need their audit information.
- Packed transform execution no longer allocates a local closure for every
  transform call.

## Preserved invariants

- Complete compatibility identity still contains boundary signature,
  direction, physical and spectral shapes, batch size, real and spectral
  dtype, and device.
- A compiled plan is local to exactly one scheduler/runtime and cannot be
  executed by another scheduler.
- The cache stores no tensors and therefore cannot extend algebraic field
  lifetime or retain state across generations.
- Timestep-local tensors continue to pass through the existing projected
  transform shape, dtype, device, and boundary checks.
- Transform batch membership, output ordering, representation-cache behavior,
  lifecycle counters, and transform counts are unchanged.
- Production Plane, Channel, and generic solver paths remain untouched.

## Qualification boundary

The candidate must be compared with the frozen Stage N.3 control using the
existing Stage N.4 trajectory and H100 qualification. All existing numerical,
final-Q, transform-count, lifecycle, memory, and performance gates remain
unchanged. Passing authorizes Stage O migration design only; it does not change
production defaults.
