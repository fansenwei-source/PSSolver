# Architecture Stage F: algebraic lifecycle and geometry dispatch

## Scope and result

Stage F extends the opt-in execution path with two contracts that were
deliberately absent from Stage E:

1. an explicit lifecycle for instantaneous algebraic fields;
2. exact dispatch of an algebraic capability to a geometry-specific solver.

It then qualifies both contracts with a coupled scalar canary.  It does not
migrate Beris--Edwards, Stokes, pressure, Plane or Channel production code.  It
does not alter the existing integrator, transform backend, numerical defaults,
benchmark scripts, checkpoints or metadata formats.

## F1: algebraic-field lifecycle

`AlgebraicSystemSpec` declares:

- a stable system and capability name;
- the algebraic components produced by the solve;
- the evolved components on which the solve depends;
- finite JSON-compatible parameters;
- the update phase.

Stage F supports exactly one phase:

```text
evolved state q^n is available
    -> solve every algebraic system a^n = A(q^n)
    -> evaluate explicit RHS N(q^n, a^n)
    -> advance evolved state to q^(n+1)
```

The implementation follows the existing semi-implicit integrator semantics.
Consequently, immediately after a timestep the stored algebraic fields still
represent `a^n` while evolved fields represent `q^(n+1)`.  They become current
again at the next pre-RHS update.  The experimental runtime exposes
`synchronize_algebraic_for_observation()` for snapshots or diagnostics that
require `A(q^(n+1))` without advancing time.  Such an observation sync does not
change the timestep or spectral-refresh clocks; the next step may intentionally
repeat the solve.

Initial algebraic fields are synchronized during runtime construction before
the first physical RHS evaluation.  `ExperimentalModelRuntime.reset()` also
rebuilds the initial algebraic state and restores the pre-RHS freshness marker.

The Stage F adapter requires:

- every algebraic field to have exactly one owning system;
- unique system and output names;
- dependencies to be evolved components only;
- exact physical-state input mappings;
- exact spectral-output mappings, shapes, dtype and device.

Algebraic-to-algebraic dependency graphs, iterative coupling between systems,
post-step constraints and multirate updates are intentionally deferred.

## F2: geometry-specific solver dispatch

`GeometrySolverRegistry` resolves the pair

```text
(exact geometry class, requested algebraic capability)
```

to one registered implementation factory.  Resolution is exact.  There is no
global default and no implicit fallback from Channel to Plane, from a bounded
geometry to a periodic implementation, or in the opposite direction.  The
stable geometry name is independently checked and recorded as provenance.
Missing and duplicate registrations fail before timestep execution.

The physical model sees only `AlgebraicSystemSpec`; it neither chooses nor
imports a concrete solver.  A registered implementation receives a restricted
`AlgebraicSolverContext` containing mathematical grid data, component
Laplacian eigenvalues, projected forward transformation, and resolved boundary
signatures.  This solver-only context is not passed to the physical model and
does not expose legacy `Fields` or `SpectralSolver`.

Each resolved system records the requested specification and exact dispatch
registration in runtime metadata.  Mutating the construction registry after a
runtime is built cannot replace its already-resolved solver objects.

## F3: coupled canary

`DiffusionHelmholtzCouplingModel` declares

```text
partial_t phi = D * Laplacian(phi) + c * response

(shift - length_sq * Laplacian) response = phi
```

`phi` is evolved and `response` is algebraic.  The model declares the
`scalar_helmholtz` capability but is independent of its implementation.

The canary registry deliberately contains two distinct registrations:

- `periodic_diagonal_helmholtz` for `PeriodicBox`;
- `plane_mixed_basis_diagonal_helmholtz` for `PlaneSlab`.

Both exploit diagonal Laplacian eigenvalues, but the dispatch identity and
resolved transform bases remain geometry-specific.  The scalar Helmholtz
factory also requires its source and output components to share one resolved
spectral basis.  `RectangularChannel` is deliberately unregistered, proving
that the dispatcher will not silently reuse the Plane implementation.

These canary solvers are experimental qualification fixtures.  They are not
the future Stokes interface and do not claim to solve pressure, a saddle
system, a zero mode or incompressibility.

## Legacy bridge and isolation

The only runtime bridge remains `pssolver.experimental`.  For a model with
algebraic fields it:

1. validates the algebraic systems against the frozen field plan;
2. resolves every capability against the supplied geometry registry;
3. constructs the unchanged legacy solver and projector;
4. declares evolved and static storage;
5. installs a spectral algebraic adapter and projected static inverse;
6. synchronizes the initial algebraic state;
7. installs the physical explicit-RHS adapter;
8. marks the initial algebraic state current for the first timestep;
9. repeats the plan/runtime shadow comparison.

Models without algebraic fields continue through the Stage E path.  The root
`pssolver` API exports none of the Stage F contracts or builders.  Production
Plane, Channel and active-nematic sources import none of them.

## Qualification

The Stage F characterization suite verifies:

- periodic and mixed periodic/Neumann coupled trajectories against separately
  assembled legacy reference solvers;
- bitwise equality of physical state, spectral state and linear operators over
  eight semi-implicit timesteps;
- initial synchronization, documented post-step staleness and observation
  synchronization;
- reset behavior for both Stage E and Stage F models;
- distinct PeriodicBox/PlaneSlab dispatch and rejection of Channel fallback;
- immutable and finite algebraic metadata;
- rejection of missing registries, invalid dependencies and duplicate
  registrations;
- absence of new architecture imports from production paths.

Desktop CPU qualification after the final implementation:

- Stage A--F focused architecture suite: `96 passed`;
- complete repository suite: `712 passed, 8 subtests passed`.

Both runs used `PYTHONDONTWRITEBYTECODE=1` and disabled the pytest cache.  No
GPU qualification is required because Stage F does not change a qualified
production execution path.

## Boundary before Stage G

Stage F makes active-nematic migration possible, not automatic.  Before a
Beris--Edwards--Stokes driver moves to the new path, Stage G must define and
test at least:

- a multi-output saddle-solver capability for velocity and pressure;
- component-wise velocity/pressure boundary spaces;
- pressure gauge and tangential zero-mode policy as model/run choices;
- solver diagnostics without treating them as state fields;
- stateful warm starts and restart provenance;
- Plane and Channel registrations as separate implementations;
- trajectory comparisons against each fixed benchmark implementation.

Until those contracts exist, current active-nematic production drivers remain
the reference and no Stage F canary registration may be substituted for them.
