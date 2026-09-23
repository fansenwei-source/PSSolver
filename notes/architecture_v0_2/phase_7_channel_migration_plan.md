# Phase 7: Channel as the second-geometry acceptance test

Status: `P7_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED`.

Phase 6 proved that the Plane architecture migration preserves the frozen
Plane trajectory.  It did not validate Channel.  Phase 7 will use the existing
no-slip rectangular Channel as the second-geometry acceptance test for the
v0.2 architecture while preserving `Channel.py` and `pssolver.channel` as the
rollback oracle.

## Architectural boundary

Channel may reuse the already qualified transform primitives, declarations,
runtime state, workspaces, integrator contracts, active-nematic constitutive
helpers, checkpoint infrastructure, and provenance schemas.  It may not reuse
a Plane geometry conclusion.  The Channel Stokes/Schur implementation,
no-slip velocity parity, pressure gauge, two bounded wall axes, PCG convergence
and warm-start state all require independent qualification.

The current production oracle uses Q in periodic/Neumann/Neumann space,
velocity in periodic/Dirichlet/Dirichlet space, and pressure in
periodic/Neumann/Neumann modal space.  A uniform tangential plug mode is not a
Channel null mode, so the Plane `zero_mean`/`friction` choice is not applicable.
The pressure mean is the gauge.  These semantics must remain explicit rather
than being inferred from transform parity alone.

Phase 7 does not add strong anchoring, non-homogeneous lifting, Robin/tau
boundaries, a new physical model, or optimal control.  Those remain Phase 8 or
the independent control project.

## Slices

1. **P7.0 — inventory and oracle freeze.** Record source identities, boundary
   spaces, pressure semantics, warm-start state, output schema, diagnostics,
   and representative CPU/H100 oracle configurations.  No runtime code moves.
2. **P7.1 — run-spec decomposition.** Introduce declarative Channel run,
   geometry, Stokes, and output specifications behind a compatibility facade.
   `Channel.py` output and defaults remain unchanged.
3. **P7.2 — Channel Stokes/Schur extraction.** Move the existing modal saddle
   solver behind a Channel-specific implementation boundary.  Add manufactured
   gradient/divergence, Schur, gauge, no-slip, wall-momentum, and fixed-iteration
   tests.  Preserve the old operation order as the oracle.
4. **P7.3 — compiled Channel declarations and execution.** Bind immutable
   declarations, runtime state, bounded workspaces, algebraic pressure state,
   and a static StepProgram.  Registry/capability lookup remains construction-
   time only.  There is no silent fallback.
5. **P7.4 — opt-in production facade.** Add explicit
   `legacy_channel`/`compiled_channel_v2` selection without changing the
   default.  Freeze requested/effective identity, metadata, checkpoint, and
   restart contracts.
6. **P7.5 — local closure.** Require full CPU/package tests, one/100-step
   Q/u/p identity, bidirectional same-runtime restart, warm-start restoration,
   negative identity gates, finite residuals, and installed-wheel execution.
7. **P7.6 — H100 closure.** Run balanced representative small and production
   grids, measure timestep, PCG iteration counts, transform calls, allocated
   and reserved memory, and compare complete trajectories.  A separate long
   Channel run is required before any default-promotion discussion.

## Gate order

Every slice is fail-closed.  The order is equation-level manufactured tests,
operation-order and trajectory identity, checkpoint/restart, residual and
gauge checks, metadata/package provenance, then H100 performance and memory.
Performance evidence cannot repair a failed scientific or identity gate.

P7.0 planning does not authorize P7.1 implementation.  The first code-changing
slice requires a separate user decision after review of the inventory and
contract.  Throughout Phase 7, Plane defaults and its frozen Phase 6 evidence
remain untouched.

The machine-readable plan is
[phase_7_channel_migration_plan.json](phase_7_channel_migration_plan.json), and
the source inventory is
[phase_7_channel_inventory.json](phase_7_channel_inventory.json).
