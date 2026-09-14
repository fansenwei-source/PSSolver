# Architecture Stage O: Plane production-migration design

## Decision and authority

Stage N.4.1 passed its bounded H100 qualification at commit
`1c1af17dcdaebbc77e125f7a04e464828afd1e56`.  Relative-L2 errors remained at
floating-point roundoff, final-Q hashes, transform counts, materialization
lifecycle, and GPU memory matched the frozen Stage N.3 control, and the
candidate/control mean timestep ratio was `0.986096952` against a fixed `1.03`
limit.

That evidence authorizes a production-migration **design**.  It does not
authorize changing a production path or default.  Stage O therefore defines a
bounded Plane-only migration with a retained rollback oracle.  Current
production code, benchmark `develop`, Channel, and the generic solver remain
unchanged in this stage.

## Scope

The first migration target is exactly the Plane/slab Beris--Edwards model with
complete nematic-stress quasistatic Stokes flow currently assembled by
`Plane_beris_edwards_stokes.py`.  It does not cover:

- `RectangularChannel` or its pressure/zero-mode implementation;
- `PeriodicBox` as a separate optimized geometry;
- the generic `SpectralSolver` public construction path;
- new physical boundary conditions or anchoring models;
- different equations, initial conditions, Q convention, dtype, dealiasing,
  transform storage, timestep scheme, or active-force convention.

Plane qualification cannot be inherited by another geometry.  Shared spectral
primitives may be reused later, but each geometry keeps its own plan, solver
dispatch, manufactured tests, and performance qualification.

## Dual runtime paths

Migration uses one explicit enum-like authority rather than a cascade of
booleans:

1. `legacy_production` remains the default and rollback oracle;
2. `separated_canary` is opt-in and uses the N.4.1 execution policy.

The resolved run specification owns this choice.  The paths cannot be selected
simultaneously, and no lower model, geometry, scheduler, or workflow layer may
reinterpret it.  A backend marker is additive metadata; it must not replace or
silently rewrite scientific metadata.

The separated implementation should be imported lazily by the canary adapter.
This keeps import-time CLI behavior out of provenance tools and prevents an
omitted selector from perturbing the legacy path.

## Responsibility map

The intended dependency direction is:

```text
model -> physical BC -> geometry -> spectral plan -> backend -> runtime
      -> workflow
```

Responsibilities are assigned as follows.

- Active-nematic equations and Q convention remain in
  `pssolver.models.active_nematics`.
- Shendruk-inspired parameter resolution becomes a pure preset rather than
  module-global arithmetic in a CLI script.
- An immutable `PlaneBerisEdwardsRunSpec` becomes the single configuration
  authority for CLI and programmatic construction.
- Physical boundary conditions and component parity are expressed through the
  core boundary and Plane geometry specifications.
- The immutable `SpectralPlan` owns the resolved FFT/DCT/DST, storage, dtype,
  mask, and retained-mode decisions.
- A Plane runtime factory owns the explicit legacy/canary choice; neither model
  equations nor the generic solver owns that choice.
- N.4.1 policy, generation lifetime, and transform schedulers remain the
  separated canary's execution mechanism.
- A shared workflow eventually owns stepping, spectral-refresh phase,
  observations, diagnostics, completion order, and provenance.
- Checkpoint formats are not assumed compatible.  Cross-format loading remains
  forbidden until a versioned adapter has been implemented and qualified.

## Migration phases

### O.1: pure configuration extraction

Extract parameter resolution, physical boundary declarations, and immutable
run configuration without changing runtime construction.  The production CLI
continues to run only the legacy path.  Existing defaults and validation errors
remain characterized, canonical scientific dry-run metadata remains equal, and
a short legacy trajectory remains byte-identical.

### O.2: opt-in runtime adapter

Introduce a narrow runtime protocol and two Plane adapters.  Omitted selection
still resolves to `legacy_production`; `separated_canary` requires an explicit
choice and is recorded in metadata.  Mixed configuration authorities fail
before solver construction.  Channel and generic solver imports remain
unchanged.

### O.3: workflow, observations, and restart boundary

Move duplicated run-loop responsibilities behind a shared workflow while
preserving the legacy Q/u/p filenames, shapes, dtypes, synchronization points,
spectral-refresh clock, and last-written `COMPLETE` marker.  Same-backend split
restart must be exact.  Unsupported legacy/canary cross-format restart must
fail before numerical execution.

### O.4: bounded production-canary qualification

Run CPU characterization plus one balanced H100 qualification.  This remains
an explicit canary and cannot alter defaults.  It must preserve short and
100-step trajectories, initial-Q identity, transform and lifecycle counts,
output schema, restart behavior, memory, and performance.

### O.5: separate promotion decision

Review all evidence, run an explicitly authorized benchmark-shaped stability
and restart canary, then either prepare a reversible default-promotion commit or
retain legacy production.  Any default change requires a separate commit,
implicit-default smoke test, and explicit user authorization.

## Frozen qualification gates

The legacy path is a characterization oracle.  Pure restructuring of it must
preserve canonical scientific dry-run metadata and a short trajectory
byte-for-byte.  The separated path may differ only at floating-point roundoff
where its qualified operation order differs.

Candidate numerical gates are fixed before execution:

- maximum relative L2 of `1e-10` for both six-step and 100-step Q/u/p
  comparisons;
- raw and demeaned pressure comparisons;
- exact initial-Q SHA-256 identity;
- finite arrays with identical shape and float64 dtype;
- same-backend split-restart equality;
- unchanged observation layout and completion ordering.

The H100 non-regression gates remain:

- three balanced pairs;
- aggregate candidate/legacy mean timestep ratio at most `1.03`;
- at least two of three paired ratios at most `1.05`;
- allocated and reserved memory ratios at most `1.03`;
- identical transform counts and algebraic lifecycle counters;
- no OOM, non-finite value, CUDA error, graph break, or compile fallback.

Passing O.4 still does not change a default.  A longer stability/restart canary
and an independent implicit-default smoke are required by O.5.

## Rollback and auditability

Until a final promotion is independently accepted, `legacy_production`
remains available and is the omitted-selector behavior.  Every migration phase
uses a fixed commit, isolated worktree, immutable input identity, predeclared
tolerances, and checksummed output.  A failure stops that phase without
modifying the reference path.

The Stage O planner validates the N.4.1 qualification artifact, hashes the
production and architecture files that define the migration boundary, and
emits only JSON design metadata.  It constructs no solver, writes no output,
and authorizes only O.1 configuration extraction as the next implementation
step.
