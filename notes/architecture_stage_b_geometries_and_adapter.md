# Architecture Stage B: concrete geometries and legacy boundary adapter

## Status and scope

Stage B builds on commit `044b4e0035d50470125be599487ac8d067c05784`.
It gives names to the tensor-product topologies already used by PSSolver and
adds one explicit compatibility boundary between physical conditions and the
current string-based runtime.

This stage remains disconnected from the production assembly and timestep
paths.  It does not alter equations, transforms, Stokes solvers, numerical
defaults, field ordering, floating-point operation order, snapshots, metadata,
or performance.

## Concrete geometry specifications

The `pssolver.geometries` package provides three immutable `GeometrySpec`
subclasses:

- `PeriodicBox`: every coordinate axis is periodic;
- `PlaneSlab`: one wall-normal axis is bounded and all tangent axes are
  periodic;
- `RectangularChannel`: one streamwise axis is periodic and all remaining
  axes are bounded.

The wall-normal axis defaults to the last domain axis, matching the current
Plane convention.  The streamwise axis defaults to the first domain axis,
matching the current Channel convention.  Both can be selected explicitly.

These classes encode topology, not wall physics.  For example, a `PlaneSlab`
does not decide between homogeneous Dirichlet and Neumann conditions.  That
choice remains attached to each field component through `BoundarySet`.

They also do not choose:

- FFT, DCT, or DST implementations;
- free-slip or no-slip Stokes closures;
- anchoring or surface-energy models;
- spectral storage or dealiasing policies;
- geometry-specific performance defaults.

## One-way legacy adapter

The `pssolver.adapters` package converts new physical boundary objects to the
string tuple accepted by the current `Fields` and transform backend:

```python
legacy = boundary_set_to_legacy(boundaries)
```

The direction is intentionally one-way:

```text
physical BoundaryCondition/BoundarySet
                  |
                  v
       legacy runtime string tuple
```

There is no adapter from arbitrary runtime strings back into physical
contracts.  This prevents old representation details from becoming the source
of truth for future models.  The mapping is centralized in the adapter rather
than exposed as a method on the core boundary objects.

## Dependency direction

The Stage B dependency graph is:

```text
pssolver.core <--- pssolver.geometries
      ^
      |
pssolver.adapters ---> current legacy boundary format
```

Neither `pssolver.core` nor `pssolver.geometries` imports the adapter.  Existing
runtime modules do not import any Stage A or Stage B package.  A future
assembler may depend on all three, but the transform backend must not depend on
physical geometry names.

## Compatibility evidence

Focused tests establish that:

- the named geometries reproduce current Plane and Channel axis topology;
- axis selection and dimensionality are validated;
- concrete geometry instances remain immutable;
- Dirichlet and Neumann wall physics both compose with `PlaneSlab`;
- every supported physical boundary maps to the current runtime label;
- adapter output is accepted unchanged by the current `Fields` interface;
- core boundary contracts no longer expose legacy conversion methods.

## Deferred work

Stage B intentionally does not add:

- a production problem assembler;
- a frozen transform or spectral plan;
- automatic model-to-runtime field allocation;
- geometry-specific Stokes dispatch;
- Plane or Channel driver migration;
- reverse parsing of legacy boundary strings;
- nonhomogeneous, Robin, tau, lifting, or surface-energy boundary plans;
- new numerical defaults.

## Stage C entry condition

Stage C may begin after focused and complete CPU tests pass and the branch is
confirmed to leave existing runtime modules untouched.  It should introduce a
read-only planning layer that resolves `ProblemSpec` into a frozen field and
transform plan, then compare that plan against existing Plane and Channel
metadata in shadow mode.  It should not yet replace production field
allocation or timestep execution.
