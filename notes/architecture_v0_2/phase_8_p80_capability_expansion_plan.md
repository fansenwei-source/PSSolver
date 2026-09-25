# Phase 8 P8.0: capability matrix and expansion plan

Status: `P8_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED`.

Baseline: `f8789461ee71391885d5e04c6906a13bc83a26b0` on
`next/pssolver-v0.2.0-architecture`.

P8.0 is a planning-only slice. It changes no equation, boundary law,
transform, runtime, timestep, checkpoint, output schema, or production
default. It authorizes neither P8.1 implementation nor an H100 job. Its
machine-readable authority is
[phase_8_p80_capability_expansion_plan.json](phase_8_p80_capability_expansion_plan.json).

## Prerequisite and decision

Phase 7 is complete. Its final closure established a tensor-free public
`Simulation` declaration, pre-allocation compilation, package-owned runtime
construction, a common runner/result protocol, and two independently
qualified application combinations. That result makes Phase 8 planning
eligible; it does not make arbitrary model--geometry--boundary combinations
executable.

Phase 8 will expand this finite capability matrix one vertical slice at a
time. It will not replace the fail-closed compiler registry with permissive
runtime guessing. A declaration may remain representable even when its
execution is unqualified; compilation must then return a structured
capability gap before tensor allocation.

## Architectural invariants

Every Phase 8 slice must preserve the following separation:

```text
model equations
      +
geometry topology
      +
physical boundary assignment
      +
numerics, time, initial condition, execution, and output
      |
  Simulation
      |
capability resolution before allocation
      |
qualified application adapter and runtime construction
      |
static timestep with no registry lookup
```

Models must not select geometries. Geometries must not select wall physics.
Boundary declarations must not name FFT, DCT, DST, lifting, tau, or a concrete
Stokes solver. Those choices belong to lowering. Application dispatch remains
outside the timestep, and there is no silent model, geometry, boundary,
runtime, or solver fallback.

Pressure gauge and wall pressure compatibility remain distinct. The Plane
tangential zero-mode policy also remains distinct from pressure gauge. A new
geometry must state both contracts rather than inherit them from the nearest
existing application.

## Current model surface

There are two qualified equation variants but only one ergonomic public model
constructor.

| Equation variant | Public status | Qualified geometry |
|---|---|---|
| `complete_stress_beris_edwards` | stable `CompleteStressBerisEdwards(...)` constructor | `plane_slab` |
| `legacy_active_force_active_nematics` | canonical internal request; no convenience constructor | `rectangular_channel` |

Internal kernel, Stokes, coupled-model, and canary classes are implementation
components, not additional `Simulation.model` choices. P8.1 may expose a
convenience constructor for the already-qualified legacy active-force
equations; it must not invent a third physical model.

## Current geometry surface

| Geometry | Topology | Ergonomic constructor | Qualified equation variant |
|---|---|---|---|
| `PeriodicBox` | every axis periodic | currently requires `DomainSpec` | none |
| `PlaneSlab` | one bounded wall-normal axis, all other axes periodic | accepts `shape` and `lengths` | complete-stress Beris--Edwards |
| `RectangularChannel` | one periodic streamwise axis, all other axes bounded | currently requires `DomainSpec` | legacy active-force active nematics |

The geometry declarations describe topology only. They do not prove that a
compatible Stokes solve, force projection, nullspace treatment, boundary
lowering, or restart path exists.

## Current boundary surface

The public field-level policies are:

- `neumann_q()`: homogeneous Neumann Q on bounded faces;
- `free_slip_velocity()`: zero normal velocity and zero normal derivative of
  tangential velocity on bounded faces;
- `neumann_pressure_compatibility()`: algebraic Neumann compatibility for the
  pressure multiplier.

Periodic continuation is induced by geometry topology and is not a separate
wall policy. The low-level core already represents periodic, homogeneous
Dirichlet, and homogeneous Neumann conditions. The Channel no-slip law is
qualified inside the Channel application but has no public
`no_slip_velocity()` policy. A core primitive is not automatically a public,
lowered, or scientifically qualified boundary choice.

Strong anchoring, prescribed nonzero Q data, nonhomogeneous lifting, finite
surface-energy anchoring, and Robin/tau lowering do not yet exist as public
capabilities.

## Frozen model--geometry matrix

| Model | `PeriodicBox` | `PlaneSlab` | `RectangularChannel` |
|---|---|---|---|
| complete-stress Beris--Edwards | planned P8.2 | qualified | planned P8.3 |
| legacy active-force active nematics | deferred | deferred | qualified |

The two current executable combinations remain:

1. complete-stress Beris--Edwards on a Plane slab, using
   `legacy_production` or `compiled_v2`;
2. legacy active-force active nematics in a rectangular Channel, using
   `legacy_channel` or `compiled_channel_v2`.

No cell in this table may be relabelled `qualified` from declaration presence
alone.

## Phase 8 slices

### P8.0 — capability matrix and scope freeze

Record the current public surface, qualified pairs, missing capabilities,
architectural invariants, implementation order, gate order, and authorization
boundary. This document and its JSON companion complete P8.0. No runtime code
changes.

### P8.1 — existing capability public surface and catalog

Expose only capabilities that already have a qualified implementation:

- an ergonomic constructor for the existing legacy active-force equation
  declaration;
- `shape`/`lengths` convenience forms for `PeriodicBox` and
  `RectangularChannel` without changing their canonical identity;
- a public `no_slip_velocity()` policy matching the qualified Channel wall
  law;
- immutable discovery of available declarations and qualified executable
  combinations.

Discovery must distinguish “declarable” from “qualified and executable.” P8.1
adds no new physics and requires local package/API/identity closure, not an
H100 job, unless implementation unexpectedly touches a hot runtime path.

### P8.2 — complete-stress Beris--Edwards in a periodic box

This is the first new executable orthogonal combination. It is deliberately
wall-free so that it tests model reuse, geometry dispatch, a fully periodic
Stokes lowering, pressure gauge, and zero-wave-number velocity policy without
mixing in new anchoring physics.

P8.2 must inventory and manufacture-test the periodic Stokes path before
registering exactly:

```text
complete_stress_beris_edwards + periodic_box
```

It then requires finite one-step and trajectory results, checkpoint/restart
identity, negative identity gates, transform-count provenance, H100
performance, and peak-memory evidence.

### P8.3 — complete-stress Beris--Edwards in a rectangular Channel

P8.3 reuses the Channel topology, no-slip velocity law, pressure gauge,
Schur/PCG solver contract, and warm-start state. It does not reuse a Plane
conclusion for bounded-axis complete-stress derivatives or force projection.
Those operators require their own manufactured tests and residual checks.

Only after those gates pass may the compiler register:

```text
complete_stress_beris_edwards
  + rectangular_channel
  + neumann_q
  + no_slip_velocity
```

The existing legacy Channel model and both application defaults remain
unchanged.

### P8.4 — generic nonhomogeneous Dirichlet lifting and strong Q anchoring

Strong anchoring is represented as prescribed physical Q data, not as a DCT
or DST name. The first lowering uses

```text
Q = Q_lift + Q_homogeneous,
Q_homogeneous = 0 on the prescribed wall,
```

with a static, time-independent lifting. The generic boundary value contract
belongs below the active-nematic convenience layer. Planar and homeotropic Q
policies are convenience constructors over that contract.

The first scientific slice is a Plane slab with strong Q anchoring. It must
verify the wall value, the lifted bulk equation, manufactured convergence,
restart identity including lifting provenance, and H100 non-regression before
any Channel extension.

### P8.5 — finite anchoring Robin/tau pilot

Finite surface-energy anchoring is not strong Dirichlet anchoring with a loose
tolerance. It requires an explicit Robin or tau operator contract and
manufactured wall-residual convergence. P8.5 is separately authorized only
after P8.4 closes; its failure or deferral must not be hidden by a fallback to
strong anchoring.

### P8.6 — Phase 8 closure

Freeze the expanded capability catalog, installed-package behavior,
structured rejection matrix, metadata, checkpoints, restart gates, numerical
oracles, and H100 evidence. Existing Phase 6 and Phase 7 evidence remains
read-only. Closure does not promote a compiled runtime or claim reproduction
of a paper benchmark unless a separate benchmark task supplies that evidence.

## Gate order

Every code-changing slice is fail-closed and uses this order:

1. static ownership, dependency, import, and public-API checks;
2. capability matrix and structured-rejection checks;
3. manufactured equation, operator, and boundary-residual tests;
4. unchanged Plane/Channel CPU identity controls;
5. new application one-step and finite-trajectory checks;
6. checkpoint, restart, tamper, and negative-identity gates;
7. installed-package, metadata, and provenance checks;
8. H100 performance, memory, transform-call, and runtime-identity gates;
9. a slice-specific long run only when required for the scientific claim.

Performance cannot repair a failed equation, boundary, gauge, residual,
restart, or identity gate. H100 evidence from one combination cannot qualify a
different geometry or wall law.

## Compatibility freeze

Throughout Phase 8:

- Plane defaults remain `legacy_production`;
- Channel defaults remain `legacy_channel`;
- `compiled_v2` and `compiled_channel_v2` remain opt-in;
- the two current qualified combinations remain available;
- existing equation, boundary, checkpoint, and output contracts do not
  change as a side effect of capability expansion;
- registry or capability lookup never enters the hot timestep;
- unsupported combinations fail before allocation without fallback.

## Authorization boundary

P8.0 planning is complete. P8.1 implementation, all later slices, H100 work,
production-default changes, and Phase 9 remain unauthorized until separately
requested. Curved or unstructured geometry, immersed boundaries, inertial
Navier--Stokes, arbitrary third-party plugin execution, and optimal control
are outside this Phase 8 plan.
