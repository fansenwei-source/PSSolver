# Phase 5 plan: opt-in compiled-v2 Plane Beris--Edwards runtime

Status: `P5_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED`.

Baseline: Phase 4 closure record commit
`e54db2f7e5be16d0e84b5d854e5232d6bb7fb72b` on
`next/pssolver-v0.2.0-architecture`.

Phase 5 planning is explicitly authorized by the user after Phase 4 closure.
This record does not authorize implementation.  It does not change the Plane
runtime enum, CLI, production default, checkpoint format, or any numerical
path.

## Objective

Create an opt-in Plane Beris--Edwards runtime with a new static control layer
and the existing qualified GPU data layout and kernels.  The path will be
selected as `compiled_v2`; `legacy_production` remains the default and
rollback oracle.

The first target is orchestration parity, not a simultaneous scientific
change.  It therefore retains projected semi-implicit Euler, the existing
mixed Fourier/DCT/DST bases, free-slip/free-Q boundary assignment, Stokes
zero-mode policy, dealiasing, spectral refresh, Q convention, output schema,
and all accepted execution optimizations.

## Non-goals

Phase 5 does not:

- promote `compiled_v2` or `separated_canary` to the default;
- connect Phase 4 SBDF2 to the production Plane application;
- change model equations, physical parameters, boundary laws, or geometry;
- add strong anchoring, non-homogeneous boundaries, Channel support, control,
  a symbolic PDE language, arbitrary block sizes, or variable timesteps;
- claim H100 performance or long-time scientific equivalence;
- delete compatibility facades or the legacy runtime.

## Runtime identities

The intended final selector contains exactly:

- `legacy_production`: unchanged default and rollback oracle;
- `separated_canary`: existing diagnostic path;
- `compiled_v2`: new explicit opt-in candidate.

Selection must be represented in scientific metadata, workflow metadata,
checkpoint headers, and run identity.  Requested and effective identities
must be equal.  Fallback is forbidden.

The `compiled_v2` enum member and CLI choice are introduced only in P5.5,
after the internal runtime can execute a complete local workflow.  Earlier
slices remain inaccessible from the application edge.

## Data and control boundary

The candidate reuses the production tensor shapes, field grouping, spectral
storage, projected-transform implementation, pointwise kernels, Q-gradient
cache, molecular-field path, stress-divergence path, and Plane free-slip
Stokes solver.  It does not reproduce the separated canary's per-component
dynamic scheduler.

Construction may use declarations and registries to resolve capabilities.
The compiled timestep receives only frozen tensor references, pre-bound
operators/callables, scalar coefficients, immutable stage order, persistent
state, and bounded workspace.  Inside one timestep it may not perform:

- registry or capability lookup;
- dictionary or string-based field discovery;
- JSON or metadata construction;
- CLI/configuration parsing;
- device/dtype inference from global state;
- unbounded tensor allocation;
- implicit runtime fallback.

Observation synchronization, output serialization, checkpointing, and
diagnostics remain outside the timed timestep program.

## Initial integrator boundary

The parity runtime uses projected semi-implicit Euler.  This avoids confusing
an architecture migration with an integrator change and permits exact
comparison with `legacy_production`.

Phase 4 proves that the generic architecture can represent SBDF2 history and
bounded modal blocks.  Enabling SBDF2 for Plane is a later scientific and
performance decision with its own initialization, restart, timestep-
convergence, and long-run gates.

## Slices

### P5.0: plan, inventory, and oracle freeze

Add ADR 0011, this plan, a machine-readable contract, a source inventory, and
static tests.  No numerical code changes.

### P5.1: disconnected compiled declarations

Introduce private immutable declarations for the compiled Plane field layout,
stage identities, persistent progress, representation ledger, and workspace
requirements.  Do not add a runtime selector or import them from production.

Completion record: `phase_5_p51_compiled_declarations.md` and
`phase_5_p51_compiled_declarations.json`.

### P5.2: construction-time binding and dataflow audit

Bind the exact production field groups, tensor references, projected
transforms, Q RHS kernel, static Stokes kernel, and scalar coefficients into a
private plan.  Compare its declared operation graph and tensor inventory with
the legacy oracle.  It must reject missing, duplicate, aliased, wrong-device,
wrong-dtype, wrong-shape, or boundary-incompatible bindings.

If reusing a built legacy solver retains its unused integrator/workspace and
violates the memory budget, stop and introduce a shared private assembly
factory behind a legacy byte-identity gate.  Do not silently accept duplicate
production workspaces.

Completion record: `phase_5_p52_construction_binding.md` and
`phase_5_p52_construction_binding.json`.

### P5.3: compiled Euler step program

Implement the fixed Euler stage program over pre-bound tensors and bounded
workspace.  Freeze callback position, projection, inverse transforms,
scheduled spectral refresh, static-field invalidation, failure atomicity, and
progress commit.  Compare one-step and short continuous CPU results with the
legacy runtime.

Completion record: `phase_5_p53_compiled_euler_step.md` and
`phase_5_p53_compiled_euler_step.json`.

### P5.4: observation, diagnostics, and checkpoint adapter

Connect observation synchronization, projected normal-force diagnostics,
pressure diagnostics, output views, and checkpoint identity without changing
Plane checkpoint format v1.  Reject cross-runtime or incompatible restart
before a timestep or output.  Add continuous/split restart and tamper tests.

Completion record: `phase_5_p54_observation_checkpoint.md` and
`phase_5_p54_observation_checkpoint.json`.

### P5.5: opt-in application connection

Only after P5.1--P5.4 pass, add `PlaneRuntimePath.COMPILED_V2`, parser support,
the private runtime adapter, and explicit application selection.  Omitted
selection remains exactly `legacy_production`.  Dry-run must not allocate
runtime tensors.  No fallback is allowed.

Completion record: `phase_5_p55_application_connection.md` and
`phase_5_p55_application_connection.json`.

### P5.6: local closure and Phase 6 handoff

Run the complete CPU suite, installed-wheel import/workflow smoke, local CUDA
smoke, deterministic 1/2/100-step parity, checkpoint restart, output/metadata
comparisons, negative identity gates, tensor-lifetime audit, and hot-loop
static audit.  Freeze the exact Phase 6 H100 and long-run qualification plan.

P5.6 completion may make Phase 6 planning eligible.  It does not authorize
Phase 6 execution or production-default promotion.

Completion record: `phase_5_p56_local_closure.md` and
`phase_5_p56_local_closure.json`.  The frozen, not-yet-authorized handoff is
`phase_6_plane_compiled_v2_qualification_plan.md` and its JSON companion.

## Required local oracles

The candidate must preserve or explicitly report:

- identical initial Q tensors and hashes;
- identical per-step operation order;
- Q/u/p byte identity where operation order is identical;
- otherwise pre-registered float64 relative-L2 and Linf tolerances;
- transform calls per step and spectral refresh counts;
- workspace tensor count, storage identity, and bounded allocation;
- completed-step and refresh counters;
- callback ordering and failure atomicity;
- checkpoint runtime identity and negative-gate behavior;
- output filenames, dtype, shapes, metadata, diagnostics, and COMPLETE rules;
- absence of fallback, graph breaks, NaN, Inf, OOM, and CUDA errors.

## Phase 6 boundary

Phase 6, not Phase 5, owns formal R128/R320 H100 balanced A/B, peak memory,
transform calls, 100-step Q/u/p comparisons, bidirectional restart if
supported, output/checksum provenance, and long-time benchmark evidence.
Passing Phase 6 candidate qualification still does not change the default;
implicit-default promotion requires a separate decision and commit.

The machine-readable authority is
[phase_5_contract.json](phase_5_contract.json), and the frozen baseline is
[phase_5_plane_inventory.json](phase_5_plane_inventory.json).
