# Phase 8 P8.4 planning: field-neutral Dirichlet lifting and strong Q anchoring

Status: `PASS_P8_4_PLANNING_IMPLEMENTATION_NOT_AUTHORIZED`.

Baseline: `9c0cdc25ad875a964cffafa73a12eda3e45c2244` on
`next/pssolver-v0.2.0-architecture`, after the authoritative P8.3 H100
closure.  P8.3 makes P8.4 planning eligible.  This record freezes ownership,
mathematics, the first qualified slice, and fail-closed gates; it implements
no boundary law, lifting operator, runtime, compiler registration, or default
change.

## Decision

The lifting mechanism is field-neutral.  Its mathematical variable is a
general evolved field `phi`, not nematic Q:

```text
phi = phi_lift + phi_homogeneous
phi_homogeneous = 0 on prescribed Dirichlet faces
```

The active-nematic package owns only convenience policies that turn physical
Q language into prescribed component data.  In particular, planar and
homeotropic anchoring construct the canonical five independent components of

```text
Q = (3 S / 2) (n n - I / 3)
```

through the existing `q_tensor` convention.  Core boundary, planning,
operator, runtime, and checkpoint code must not know about directors, scalar
order, traceless tensors, or active nematics.

Generality is a declaration and ownership property, not a claim that every
field and geometry becomes executable in the first implementation.  The first
qualified scientific combination remains deliberately narrow:

```text
complete_stress_beris_edwards
  + plane_slab
  + static prescribed Dirichlet Q on both z faces
  + existing free_slip_velocity
  + existing pressure compatibility
```

## Current code and the missing capability

The current architecture already has a face-aware `BoundaryAssignment` that
attaches laws to arbitrary logical components.  It deliberately rejects
nonhomogeneous conditions.  Public boundary helpers expose homogeneous
Neumann, free-slip, no-slip, and pressure-compatibility policies.  Lowering
maps periodic, homogeneous Neumann, and homogeneous Dirichlet modal kinds to
FFT, DCT, and DST.  The DST implementation can represent
`phi_homogeneous`, but no object owns prescribed values and no runtime forms
or applies `phi_lift`.

Writing a tensor into `Field` is initialization, not lifting: it neither
preserves a nonzero wall value nor transforms the bulk equation.  No old Plane
or Channel script supplies a reusable strong-Q lifting implementation.

## Contract ownership

### Generic physical declaration

`pssolver.core.boundary` will own tensor-free, field-neutral prescribed-value
identity.  The planned contract records boundary kind, static/time-independent
status, component, oriented face, value representation, finite canonical
values or content-addressed snapshot identity, and deterministic metadata.
It does not hold a Torch tensor, callable, transform name, model object, or
geometry-specific lifting algorithm.

The first implementation admits only static prescribed Dirichlet data for an
evolved field.  Algebraic, transient, and diagnostic fields; time-dependent
data; arbitrary Python callables; trainable boundary values; Robin data; and
surface energies fail before allocation.

### Public field policy

`pssolver.boundaries` will expand a field-neutral prescribed-Dirichlet policy
into component/face assignments.  It may target any registered evolved field
whose component set matches the supplied data.  Periodic faces cannot carry
prescribed values.  Lower and upper bounded faces have independent identities.

The public declaration must not mention FFT, DCT, DST, lifting, tau, a solver,
or an active-nematic model.

### Planning and lowering

`pssolver.planning` will own an immutable `LiftingRequirement` and
`StaticLiftingPlan`.  Lowering validates field role, geometry faces, component
coverage, value identity, dtype intent, and the available geometry-specific
builder.  It selects a homogeneous remainder basis separately from the
physical boundary declaration.

For the first Plane implementation, the homogeneous remainder uses the
existing periodic/periodic/DST space.  Lowering binds a static Plane lifting
builder; it does not reinterpret strong anchoring as the name `DST`.

### Geometry-aware lifting operator

A model-neutral operator materializes the static lift once on the selected
device and dtype.  The first Plane builder uses an affine wall-normal
extension between independently prescribed lower and upper wall values.  For
constant wall data this satisfies the wall values exactly.  Future spatially
varying face data require a separately qualified extension; the first slice
does not silently accept them.

The materialized lift, its homogeneous-space representation, derivative data,
and any linear-operator correction are immutable runtime state or workspace.
They are not rebuilt or allocated in every timestep.

### Q-specific convenience layer

`pssolver.models.active_nematics.boundaries` will own planned conveniences
such as `strong_planar_q`, `strong_homeotropic_q`, and `prescribed_q`.  They
validate unit directors, finite nonnegative scalar order, the canonical
`de_gennes_S_lambda_max_v1` convention, symmetric/traceless Q structure, and
the five-component ordering.  They return generic prescribed boundary data;
they do not construct transforms, solvers, or a runtime.  The convenience
layer consumes an explicit unit director or an abstract oriented-face normal
provided during composition; it must not import a concrete Plane or Channel
geometry.

## Lifted equation contract

Every physical constitutive and diagnostic evaluation receives

```text
phi_physical = phi_homogeneous + phi_lift.
```

The evolved state is the homogeneous remainder.  For a general split equation

```text
partial_t phi = L(phi) + N(phi),
```

the static-lift equation is

```text
partial_t phi_homogeneous =
    L(phi_homogeneous)
    + L(phi_lift)
    + N(phi_homogeneous + phi_lift).
```

The implementation must represent the `L(phi_lift)` correction explicitly,
even when a particular affine constant-wall lift makes that term analytically
zero.  Merely adding the lift during output is incorrect.  A future
time-dependent lift would also require `-partial_t phi_lift`; it is outside
the first slice and must be rejected.

Initial conditions are physical-field declarations.  Construction computes
the homogeneous initial remainder, verifies prescribed wall values, and
rejects incompatible input instead of silently overwriting it.  Observations,
diagnostics, constitutive kernels, and saved snapshots expose the physical
field.  Internal integrator and checkpoint records identify the homogeneous
remainder representation explicitly.

## Restart and provenance

Checkpoint identity will include the prescribed-data identity, lifting-plan
identity, convention identity where applicable, materialized-lift SHA-256,
representation kind, geometry identity, dtype, and runtime/backend identity.
The static lift is rebuilt and hash-verified before target mutation.  The
checkpoint stores the evolved homogeneous remainder; it need not duplicate a
deterministically reconstructible static lift.

Continuous and split/restarted trajectories must be byte-for-byte identical
for physical Q, velocity, and pressure.  Negative gates cover wall-data,
lifting-plan, convention, tensor-file, runtime, backend, geometry, shape,
dtype, and finite-state tampering before any target mutation.

## Performance contract

There is no registry lookup, policy dispatch, value-provider call, or new
allocation in the timestep hot loop.  Static lift tensors and corrections are
constructed once.  Physical-field reconstruction uses preallocated workspace
and may be fused only after an unfused oracle is established.  H100 closure
must report timestep, peak allocated/reserved memory, transform calls, graph
breaks, fallback state, and numerical identity against the frozen homogeneous
Plane control.

## Planned slices

1. **P8.4.0 — planning (this record).** Freeze ownership, equations, scope,
   rejection semantics, and gates.  No implementation.
2. **P8.4.1 — generic declarations.** Add static prescribed Dirichlet data,
   field-level policy, metadata, stable identity, and structured rejection.
   Keep all active-nematic names out of the generic layer.
3. **P8.4.2 — Plane lifting operator.** Add affine static lift, homogeneous
   remainder conversion, explicit linear correction, wall-value tests, and
   independent manufactured convergence.
4. **P8.4.3 — Q conveniences and application lowering.** Add canonical planar,
   homeotropic, and explicit-Q policies and register only the first Plane
   strong-anchoring combination after CPU oracles pass.
5. **P8.4.4 — workflow and restart.** Save physical observations, checkpoint
   the homogeneous remainder and provenance, verify exact restart, and run
   negative identity/tamper gates.
6. **P8.4.5 — H100 closure.** Qualify installed-package identity, wall
   residuals, manufactured convergence, finite trajectories, exact restart,
   performance, memory, transform counts, provenance, and checksums.

Each implementation slice requires separate user authorization.  A later
slice may not reinterpret the planning record as permission to implement all
remaining work.

## Explicit non-goals

P8.4 does not add finite/Robin anchoring, tau rows, surface-energy dynamics,
time-dependent or trainable boundary data, optimal control, arbitrary curved
geometries, Channel strong anchoring, automatic fallback, runtime promotion,
or a benchmark claim.  Robin/tau finite anchoring remains P8.5.  Phase 9
remains unauthorized.

## Authorization boundary

P8.4 planning is complete.  P8.4.1 implementation, H100 submission, P8.5,
Phase 9, production-default changes, and compiled-runtime promotion remain
unauthorized until separately requested.
