# ADR 0011: Plane compiled-v2 uses a static control layer over qualified kernels

## Status

Accepted for Phase 5 planning.  Implementation is not yet authorized.

## Context

The production Plane Beris--Edwards application is fast and scientifically
qualified, but its orchestration still descends from the legacy monolithic
solver.  The separated canary demonstrates clearer ownership, yet its
component-wise dynamic dispatch is not an acceptable production data path.
Phase 4 qualified state/history contracts, constant-step SBDF2, bounded
two-component modal solves, and an H100 combined canary; it did not connect
those reference capabilities to Plane.

The target architecture requires a new Plane path without discarding the
existing GPU layout, mixed-basis transforms, projected dealiasing, pointwise
Beris--Edwards kernels, or free-slip Stokes implementation.

## Decision

Phase 5 will design an opt-in runtime identity named `compiled_v2`.  It will
use a construction-time compiler to bind a fixed Plane execution plan and a
bounded workspace over the already qualified high-performance tensor layout
and numerical kernels.  The timestep loop may not perform registry lookup,
JSON parsing, CLI interpretation, string-based field discovery, or dynamic
dependency resolution.

The first parity target keeps projected semi-implicit Euler and the exact
legacy scientific configuration.  It does not simultaneously activate
SBDF2, change a boundary condition, alter a zero-mode rule, or introduce a new
checkpoint format.  Phase 4's second-integrator capability remains available
for a later, separately qualified Plane decision.

`legacy_production` remains the default.  `separated_canary` remains a
diagnostic architecture path and is not the implementation basis of
`compiled_v2`.  Runtime selection is explicit and fail-closed; no fallback
between the three identities is permitted.

Phase 5 is additive.  A new selector is exposed only after an end-to-end
compiled runtime exists.  Until then, internal plan/state/workspace objects
remain private and disconnected.  Formal H100 performance, long trajectory,
restart, output, and implicit-default promotion gates belong to Phase 6.

## Consequences

- Existing production kernels and data layouts remain the numerical oracle.
- The new control layer can become easier to inspect without paying the
  separated canary's dynamic-dispatch cost.
- Plane parity is isolated from the separate question of using SBDF2 in a
  production active-nematic run.
- Any extraction that changes the legacy hot path requires its own byte-
  identity and non-regression gate; an additive implementation is preferred.
- Phase 5 completion authorizes Phase 6 qualification, not default promotion.
