# Phase 5 P5.1: disconnected compiled Plane declarations

Status: `PASS_P5_1_DISCONNECTED_COMPILED_DECLARATIONS`.

Baseline: P5.0 planning record commit
`886a48b64b74aba74680583dd0d3d15074bac28e`.

The user explicitly authorized the next Phase 5 slice.  P5.1 adds a private,
immutable, tensor-free description of the intended `compiled_v2` Plane
runtime.  It does not add the runtime selector, bind tensors, execute a
timestep, or modify either existing Plane path.

## Added declaration boundary

The direct-import-only module
`pssolver.planning.plane_compiled_v2` records:

- evolved component order `(Qxx, Qxy, Qxz, Qyy, Qyz)`;
- algebraic component order `(ux, uy, uz, p)`;
- one Neumann Q group in the Fourier/Fourier/DCT basis;
- one Neumann tangential-velocity group in that same basis;
- one Dirichlet normal-velocity group in the Fourier/Fourier/DST basis;
- one Neumann pressure-modal group in the Fourier/Fourier/DCT basis;
- the exact qualified projected semi-implicit Euler operation order;
- persistent physical/spectral Q state, progress counters, spectral-refresh
  counters, and representation ledger;
- semantic, bounded, construction-bound workspace categories;
- dynamic work forbidden inside the future compiled timestep.

The module uses only Python standard-library declarations.  It does not
import Torch or any PSSolver runtime, integrator, model, operator, solver, or
application module.  It allocates no tensors.

## Exact operation order

P5.1 mirrors the existing qualified `ProjectedSemiImplicitEulerStepProgram`:

1. `prepare_algebraic`;
2. `pre_update_callback`;
3. `explicit_rhs`;
4. `spectral_add_dt_rhs`;
5. `spectral_divide_by_denominator`;
6. `project_dynamic_spectra`;
7. `inverse_dynamic_spectra`;
8. `scheduled_spectral_refresh`;
9. `commit_progress`.

This is a declaration, not an alternative implementation.  P5.3 owns the
future executable compiled Euler program.

## Validation and rejection rules

The immutable value objects reject:

- invalid or duplicate identifiers;
- evolved/algebraic overlap;
- missing, duplicated, or role-incompatible transform-group membership;
- reordered Euler stages;
- duplicate workspace identities;
- unbounded or construction-unbound workspace categories;
- a connected runtime in the P5.1 declaration.

Static tests additionally require that importing the private module does not
import Plane runtime modules, expose the declaration through
`pssolver.planning` or package-root APIs, or add `compiled_v2` to the current
runtime selector.

Validation result:

- targeted P5.0/P5.1 and architecture-boundary tests: 24 passed;
- complete CPU suite: 1853 passed and 8 subtests passed;
- `git diff --check`: passed.

## Deliberately unchanged

P5.1 does not change:

- `PlaneRuntimePath`;
- CLI choices or default selection;
- `legacy_production` or `separated_canary`;
- tensor layout, numerical kernels, equations, boundaries, integrator,
  checkpoint format, output schema, or metadata schema;
- production imports or hot paths;
- Phase 6 authorization or production-default status.

## Next boundary

P5.2 is locally eligible but not yet implemented.  It may bind the declared
fields, coefficients, operators, and workspace to the exact production
tensors and audit aliases, shapes, dtypes, devices, boundary compatibility,
and storage ownership.  It must remain private and disconnected from the
application selector.

The machine-readable authority is
`phase_5_p51_compiled_declarations.json`.
