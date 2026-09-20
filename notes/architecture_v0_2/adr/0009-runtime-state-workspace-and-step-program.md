# ADR 0009: runtime state, workspace, and step-program migration

Status: accepted for Phase 3.

## Context

The qualified v0.1.2 lineage stores layout, physical values, spectral values,
linear operators, and transform behavior in `Fields`.  Integrator progress is
stored separately, while checkpoint code reconstructs one logical state from
both objects.  Static fields and several algebraic caches are reconstructible,
but their lifetime is not represented by one common contract.  Three
integrators also encode nearly the same timestep order independently.

Replacing these objects in one change would combine ownership migration,
floating-point reordering, checkpoint evolution, and hot-loop optimization.
That is not an auditable migration.

## Decision

Phase 3 uses a strangler migration with four explicit concepts:

- `RuntimeState` is the sole eventual owner of evolved physical and spectral
  tensors, time/integrator progress, representation validity, and declared
  persistent algebraic tensors.
- `WorkspacePlan` describes bounded generation-local storage.  A runtime
  workspace owns preallocated buffers, but their values are invalid outside
  the generation in which they are published.
- `StepProgram` freezes the ordered numerical operations and direct callables
  before entering the timestep loop.
- the integrator requests execution of that program and commits progress only
  after every required operation succeeds.

The first `RuntimeState` implementation is disconnected.  It holds supplied
tensors by identity and performs no allocation, clone, conversion, transform,
or import-time registration.  It is imported directly from
`pssolver.execution.state` and is not a package-root stable API in Phase 3.

## Persistent state boundary

Persistent state consists of:

- evolved physical and native spectral representations;
- completed-step count, physical time, spectral-refresh interval and phase;
- explicit representation-generation validity;
- solver state that is mathematically persistent across steps, such as an
  authorized warm start.

The diagonal linear operator, transform resources, masks, wavenumbers, and
semi-implicit denominator belong to the immutable execution plan.  Gradients,
molecular fields, stresses, forces, reconstructed flow fields, diagnostics
scratch, and transform scratch belong to workspace unless a model explicitly
declares them persistent.

## Representation rule

A logical update advances exactly one representation generation.  A transform
synchronizes the other representation to that generation without advancing
time.  At least one representation must always be current.  Debug/reference
execution checks these transitions; the final qualified hot loop may lower
them to fixed calls after the program is validated.

## Compatibility rules

- `legacy_production` remains the production default until an independent
  qualification explicitly promotes another path.
- `separated_canary` remains opt-in.
- Plane checkpoint format v1 remains readable and writable without schema or
  byte-layout changes during the first connection stages.
- `SpectralSolver`, `Fields`, and current integrator public attributes remain
  compatibility facades until their consumers have migrated.
- no Phase 3 connection may change equations, boundary conditions, transform
  bases, dealiasing, zero-mode handling, or floating-point operation order.
- the disconnected contracts cannot be imported by the existing runtimes
  until a canary connection stage is separately frozen and qualified.

## Timestep order oracle

For the projected Plane path the frozen order is:

1. consume a current reconstructed algebraic state or recompute it;
2. invoke the pre-update callback;
3. compute the explicit right-hand side;
4. add `dt * rhs` to evolved native spectra;
5. divide by the precomputed semi-implicit denominator;
6. project evolved spectra;
7. inverse-transform evolved fields to physical space;
8. if due, rebuild and project spectra from physical space;
9. commit completed-step and refresh progress.

Callback timing and scheduled refresh behavior are observable contracts.

## Rollout

P3.0 freezes this inventory and oracle.  P3.1 adds disconnected state.
P3.2 adds bounded workspace contracts.  P3.3 extracts a compatible step
program.  P3.4 connects only `separated_canary`.  P3.5 may connect the legacy
production facade after byte-exact restart and trajectory gates.  P3.6 closes
the phase with package, CPU, restart, H100 performance, memory, and numerical
qualification.

## Consequences

Ownership becomes testable before production data moves.  The migration needs
temporary facades, but each stage has one numerical authority and a reversible
connection point.  Performance work remains possible because the target is a
pre-bound packed program rather than a mapping-driven hot loop.
