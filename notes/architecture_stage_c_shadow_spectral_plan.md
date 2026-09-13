# Architecture Stage C: frozen spectral plan and shadow comparison

## Status and scope

Stage C builds on commit `898c2b1d3f1326d0c0fd297399515936e43877a7`.
It converts a validated `ProblemSpec` into an immutable, tensor-free
`SpectralPlan` and can compare that plan with metadata produced by the current
runtime.

The plan is not installed into `SpectralSolver`, `Fields`, `PDEModel`, the
projector, a Stokes solver, or a timestep loop.  Existing production execution
therefore remains unchanged.

## Planning pipeline

```text
ModelProtocol + GeometrySpec + BoundarySet + NumericsConfig
                             |
                             v
                        ProblemSpec
                             |
                pure, tensor-free assembly
                             |
                             v
                        SpectralPlan
                             |
                 optional shadow adapter
                             |
                             v
          current backend / projector / Fields metadata
```

Assembly has no device, PyTorch tensor, transform matrix, cache, executor, or
field value.  It is safe to run before selecting CPU or GPU execution.

## Frozen plan contents

`DomainAxisPlan` records, for each axis:

- coordinate index and name;
- periodic or bounded topology;
- physical and spectral storage sizes;
- physical length;
- whether the axis is Hermitian-packed.

`ComponentTransformPlan` records, for each scalar component:

- its logical field and role;
- physical boundary conditions;
- resolved FFT, DCT, or DST family;
- the number of modes retained by dealiasing;
- the transform extent actually computed by full or truncated execution;
- its legacy runtime storage index, or no index for diagnostics.

`FieldPlan` preserves logical field grouping.  `SpectralPlan` preserves field
declaration order while exposing a separate runtime storage order:

1. evolved components;
2. algebraic components;
3. diagnostic components are declared but not runtime-stored.

This mirrors the current dynamic/static layout without retaining the
misleading term `static` in the architecture contracts.

## Dealiasing and storage semantics

Planning distinguishes four quantities that must not be conflated:

1. physical grid shape;
2. spectral storage shape;
3. modes retained by the dealiasing mask;
4. axis extents actually computed by projected transforms.

Periodic axes cannot use the current contiguous real-basis truncation, so a
truncated projected transform still computes their full stored FFT extent.
DCT and DST axes may compute only the retained low-mode block.  Hermitian
packing changes storage size independently of dealiasing.

The planner uses exact rational cutoff arithmetic for `two_thirds` and
`cubic_half`.  Characterization tests compare the resulting counts with the
qualified runtime projector on odd and even grids, full-complex and Hermitian
storage, and FFT/DCT/DST component layouts.

## Shadow comparison

`compare_spectral_plan_to_runtime` checks structural metadata from an existing
backend and can additionally inspect current projector and `Fields` metadata.
It reports deterministic path-based mismatches and provides `require_match()`
for validation gates.

The comparison checks:

- shape, physical lengths, precision, storage and Hermitian axis;
- transform execution order;
- per-component legacy boundary tuples and transform families;
- axis-mode, Laplacian and squared-wavenumber shapes;
- retained and computed dealiasing sizes;
- flattened component order and dynamic/algebraic counts.

It never modifies solver fields, coefficients, equations, or execution paths.
Querying current backend/projector metadata may populate their existing
internal metadata caches; this is the only permitted runtime-side effect.

## Fail-fast invariants

The plan validates:

- ordered contiguous axes and storage indices;
- unique logical field and component names;
- component dimension against domain dimension;
- FFT only on periodic topology and DCT/DST only on bounded topology;
- retained and computed extents against spectral storage sizes;
- Hermitian packing against the numerical configuration;
- finite, JSON-compatible model parameter provenance.

Assembly consumes the snapshot already held by `ProblemSpec`.  Mutating the
original model after constructing the problem cannot change the generated
plan.

## Deferred work

Stage C does not add:

- executable transform objects or tensor allocation;
- model operator execution contracts;
- automatic linear-operator construction;
- geometry-specific Stokes dispatch;
- replacement of `PDEModel.add_dynamic_field` or `add_static_field`;
- production driver migration;
- a compatibility promise for the plan schema beyond this refactor branch;
- nonhomogeneous or Robin boundary planning.

## Stage D entry condition

Stage D may begin after Stage C focused and complete CPU validation passes.
It should add an opt-in compatibility assembler that materializes the current
legacy field declarations and backend configuration from a frozen plan, then
compare short CPU trajectories with manually assembled reference solvers.  It
must remain behind an explicit experimental entry point and must not change
Plane, Channel, or generic solver defaults.
