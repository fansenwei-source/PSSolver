# PSSolver v0.1.2 API surface inventory

This is a factual inventory of the API surface at the frozen v0.1.2 release,
commit `4fa614616e9d67c98be52c150eaf303a0c80b5c1`.  It classifies existing names;
it does not broaden the supported scientific scope.

## Supported application API

The qualified public application surface is:

- `pssolver.applications.run_plane_beris_edwards`;
- `pssolver.configuration.create_plane_beris_edwards_run_spec`;
- `pssolver.configuration.PlaneBerisEdwardsRunSpec`;
- `pssolver.workflows.PlaneWorkflowResult`, returned by the callable
  application;
- installed CLI `pssolver-plane-beris-edwards`;
- compatibility CLI `python -m Plane_beris_edwards_stokes`.

The supported numerical runtime selected by an omitted selector is
`legacy_production`.  `separated_canary` is opt-in and is not a supported
production replacement.

The application contract includes:

- mixed periodic/DCT/DST Plane execution;
- the documented Beris--Edwards--Stokes--Brinkman equations and Q convention;
- float64 qualification;
- Q/u/p snapshot schemas;
- diagnostics, metadata, restart/checkpoint, and atomic `COMPLETE` ordering.

## Compatibility-public root imports

The following names are exported by `pssolver.__all__` in v0.1.2.  They must
remain importable during mechanical module extraction, but new architecture
code should not treat their current implementation modules as permanent.

```text
__version__
SpectralSolver
Fields
Parameters
DEFAULT_TRANSFORM_GROUP_INDEXING
TRANSFORM_GROUP_INDEXING_MODES
PDEModel
SemiImplicitEulerIntegrator
CONVENTION_NAME
IDEAL_LOOP_MODE_AXES
classify_ideal_loop
convention_metadata
ideal_loop_axis_angles
BasisAwareSpectralProjector
DEALIAS_RULE_FRACTIONS
DEFAULT_DEALIAS_RULE
DEFAULT_PROJECTED_TRANSFORM_EXECUTION
DEFAULT_PERIODIC_TRANSFORM_EXECUTION
DEFAULT_SPECTRAL_STORAGE
DEFAULT_TRANSFORM_EXECUTION_ORDER
FreeSlipModalStokesSolver
TensorProductTransformBackend
PROJECTED_TRANSFORM_EXECUTION_MODES
PERIODIC_TRANSFORM_EXECUTION_MODES
SPECTRAL_STORAGE_MODES
projected_common_basis_stress_divergence
projected_distortion_stress_divergence
prepare_new_run_directory
write_run_metadata
SimulationSnapshot
REPRESENTATIVE_ORDERED_S_DEFINITION
apply_snapshot_to_solver
load_snapshot
representative_ordered_S
require_distinct_output_directory
```

Compatibility means that import paths, signatures, defaults, and documented
behavior are preserved during v0.2 migration.  It does not make every internal
attribute of these classes a stable v1 API.

### Direct `pssolver.transforms` compatibility surface

`pssolver.transforms` has no explicit `__all__` in v0.1.2.  Phase 0 therefore
freezes the following intentionally defined or deliberately imported names as
the conservative direct-import surface for the Phase 1 facade:

```text
DEFAULT_DEALIAS_RULE
DEFAULT_TRANSFORM_EXECUTION_ORDER
DEFAULT_PROJECTED_TRANSFORM_EXECUTION
PROJECTED_TRANSFORM_EXECUTION_MODES
DEFAULT_SPECTRAL_STORAGE
SPECTRAL_STORAGE_MODES
DEFAULT_PERIODIC_TRANSFORM_EXECUTION
PERIODIC_TRANSFORM_EXECUTION_MODES
DEALIAS_RULE_FRACTIONS
TransformMetadata
TensorProductTransformBackend
BasisAwareSpectralProjector
projected_common_basis_stress_divergence
projected_distortion_stress_divergence
FreeSlipModalStokesSolver
BoundedAxisPlanKey
DenseBoundedAxisExecutionPlan
build_dense_orthonormal_matrix
```

Imported implementation modules and helpers such as `math`, `torch`,
`functional`, and `dataclass` are incidental module attributes, not PSSolver
APIs.  Phase 1 characterization will verify the list, signatures, root
re-exports, class-module introspection, and checkpoint behavior before moving
the first implementation.

## Provisional architecture API

The following namespaces contain typed contracts that are suitable for
repository-internal use and architecture experiments, but remain provisional
until at least two production consumers validate them:

- `pssolver.core`;
- `pssolver.geometries`;
- `pssolver.planning`;
- `pssolver.execution`;
- `pssolver.backends`;
- `pssolver.models.active_nematics`;
- `pssolver.configuration` beyond the supported Plane constructors;
- `pssolver.runtime`;
- `pssolver.workflows` beyond the supported Plane result;
- `pssolver.control`.

The active-nematic package intentionally exposes useful equation, Q-tensor,
initial-condition, adapter, and diagnostic building blocks.  Source
compatibility should be preserved where practical, but these names have not
yet earned a frozen v1 model protocol.

## Internal and experimental surface

The following are implementation or research surfaces and are not production
public APIs:

- `pssolver.experimental` and every Stage-specific implementation;
- `pssolver.diagnostics` implementation helpers;
- private backend, cache, workspace, and packing details;
- benchmark and profiling programs;
- `scripts_plane`, `scripts_channel`, tests, and validation helpers;
- the unsupported Channel production path;
- root historical research scripts other than the compatibility Plane CLI.

An internal name appearing in a wheel does not promote it to supported status.

## Output and checkpoint compatibility

For the supported Plane application, Q/u/p filenames, array shapes, dtypes,
diagnostics, completion ordering, and same-backend restart behavior are part of
the v0.1.2 compatibility boundary.

Internal storage indices, transient workspaces, transform caches, compiled
graphs, and Python class module names are not permanent checkpoint APIs.  Phase
1 must nevertheless audit class `__module__` changes because they may affect
introspection, pickle, or provenance even when numerical arrays are identical.

## Deprecation policy for v0.2 work

- Do not delete or silently redirect a compatibility-public name.
- Add a canonical replacement before announcing deprecation.
- Preserve the old path through a thin facade for at least two minor releases.
- State whether a migration changes only implementation identity or also
  changes scientific/discretization meaning.
- Remove a facade only after repository and installed-wheel tests cover the
  replacement.
