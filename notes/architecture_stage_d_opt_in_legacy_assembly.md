# Architecture Stage D: opt-in legacy runtime assembly

## Status and scope

Stage D builds on commit `228f629beacba458fa0179e607fae50d44656407`.
It introduces an explicit experimental bridge from a frozen `SpectralPlan` to
the current `SpectralSolver`, projector, and `PDEModel` field-declaration API.

The bridge lives under `pssolver.experimental`.  It is not re-exported by the
root `pssolver` package and is not imported by existing production modules or
drivers.  Plane, Channel, Periodic, benchmark, and generic solver defaults are
unchanged.

## Two-phase compatibility assembly

The Stage D path is deliberately split into a pure phase and a runtime phase:

```text
SpectralPlan
     |
     | materialize_legacy_assembly (pure, tensor-free)
     v
LegacyAssemblySpec
     |
     +--> create_legacy_solver
     +--> create_legacy_projector
     +--> declare_legacy_fields
```

`LegacyAssemblySpec` contains immutable constructor declarations:

- `LegacyBackendSpec`: shape, lengths, precision, transform order, spectral
  storage, and Hermitian axis;
- `LegacyProjectorSpec`: dealias rule and full/truncated execution;
- ordered `LegacyFieldDeclaration` entries with current string boundary tuples
  and dynamic/static compatibility roles;
- an explicit list of diagnostic components omitted from current runtime
  storage.

Materialization does not import initial data or construct tensors.

## Explicit runtime inputs

`create_legacy_solver` requires runtime-only values such as `dt`, device, and
batch size.  It constructs a new empty solver and immediately shadow-checks
its backend against the source plan.

`create_legacy_projector` separately constructs and shadow-checks the current
dealiasing projector.  It does not silently attach the projector to an
integrator or nonlinear model.

`declare_legacy_fields` requires exact mappings for every evolved component:

- physical initial values;
- spectral linear operators.

It validates mapping keys, tensor types, physical/spectral shapes, solver
identity, and existing declarations before changing the new solver.  The
function adds evolved fields through `add_dynamic_field` and algebraic fields
through the current `add_static_field` compatibility API.

It intentionally does not call `solver.build()`.  Callers must first install
the appropriate nonlinear model, algebraic compute model, or specialized
integrator.  This prevents the compatibility layer from inventing a physical
execution contract that Stage A deliberately deferred.

## Diagnostics

The current runtime has no first-class diagnostic field role.  A plan that
contains diagnostics is rejected by default.  A caller may explicitly set
`allow_unstored_diagnostics=True`; their names are then preserved in the
assembly metadata but no runtime storage is allocated for them.

This acknowledgement makes omission auditable and prevents a diagnostic from
silently becoming a legacy static field.

## Current-runtime limitations made explicit

Materialization rejects a plan with no evolved component because the current
`PDEModel` and semi-implicit integrator require at least one dynamic field.
Only homogeneous periodic, Dirichlet, and Neumann conditions can reach the
legacy adapter.  Nonhomogeneous, Robin, tau, lifting, and surface-energy
conditions remain outside this bridge.

The names `dynamic` and `static` appear only in compatibility metadata and
calls to the old API.  The architectural source of truth remains `EVOLVED`,
`ALGEBRAIC`, and `DIAGNOSTIC`.

## Equivalence evidence

Focused CPU characterization constructs independent manual and plan-assembled
solvers for:

- a fully periodic box;
- a mixed FFT/DCT plane slab with Hermitian storage;
- a mixed FFT/DST/DCT rectangular channel.

Each pair receives identical physical initial data and independently generated
linear operators.  Initial spatial arrays, spectral arrays, linear operators,
and five semi-implicit timesteps are required to be bitwise identical.

Negative tests verify:

- missing, extra, and non-string field keys are rejected before mutation;
- invalid tensor shapes leave the solver empty;
- an already-declared or structurally mismatched solver is rejected;
- diagnostics require explicit acknowledgement;
- plans without evolved components cannot be materialized;
- invalid timestep and batch values are rejected;
- the experimental functions are absent from the production package API.

## Rollback and risk boundary

Stage D changes no existing runtime file.  Rolling it back consists only of
removing the `pssolver.experimental` package, its focused tests, and this note.
No data format, checkpoint, metadata schema, numerical default, or production
driver depends on it.

## Stage E entry condition

Stage E may begin after focused and complete CPU tests pass and repository
isolation is confirmed.  It should define the smallest execution-facing model
contract needed by two simple canary models, then migrate a non-benchmark
example through the experimental assembler.  Active-nematic Plane/Channel
drivers and geometry-specific Stokes dispatch should remain unchanged until
that contract has demonstrated trajectory equivalence and a clean rollback
path.
