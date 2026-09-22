# Phase 5 P5.2: construction binding and dataflow audit

Status: `PASS_P5_2_CONSTRUCTION_BINDING_AND_DATAFLOW_AUDIT`.

Baseline: P5.1 commit
`7b602c01d4023abfd23dec9a1940348f6fdb9cbc`.

The user explicitly authorized the next Phase 5 slice.  P5.2 binds the P5.1
declaration to an untouched `legacy_production` Plane construction and audits
the resulting object graph.  It remains private and disconnected: no runtime
selector, application import, executable compiled timestep, or fallback was
added.

## Private binding boundary

`pssolver.runtime.plane_compiled_v2_binding` accepts exactly:

- one resolved `PlaneBerisEdwardsRunSpec` selecting `legacy_production`;
- the qualified `SpectralSolver` built from that specification;
- the exact `BasisAwareSpectralProjector` returned by the production builder.

It returns a frozen `PlaneCompiledV2BindingPlan` containing references to:

- packed evolved and algebraic physical/spectral storage;
- the linear operator, semi-implicit denominator, and activity tensor;
- the explicit Q RHS and complete-nematic-force Stokes models;
- pre-bound algebraic preparation, explicit RHS, projection, inverse, and
  spectral-refresh operations;
- scalar coefficient values and their configuration authorities;
- semantic and actual execution transform groups;
- the existing runtime state and zero-byte legacy workspace.

The plan adopts references only.  It does not execute a timestep or allocate
a persistent tensor/workspace.

## Semantic groups versus execution groups

P5.1 records four semantic physical roles:

- five Neumann Q components;
- two Neumann tangential velocity components;
- one Dirichlet normal velocity component;
- one Neumann pressure component.

P5.2 separately records the exact production batching:

- dynamic Q indices `[0, 1, 2, 3, 4]`, contiguous slice;
- static Neumann indices `[5, 6, 8]` for `(ux, uy, p)`, advanced indexing;
- static Dirichlet index `[7]` for `uz`, contiguous slice.

This distinction prevents the architecture record from confusing scientific
roles with the batching performed by `PDEModel.update_static_fields`.

## Construction gates

Binding rejects:

- missing, duplicate, reordered, or wrongly indexed fields;
- wrong dynamic/static registration counts or ordering;
- incompatible domain, batch size, timestep, dtype, spectral dtype, device,
  storage layout, Hermitian axis, dealias rule, or projected-transform mode;
- missing, mixed, or boundary-incompatible transform groups;
- unexpected aliases among physical, spectral, linear-operator,
  denominator, and activity ownership roots;
- copied rather than zero-copy RuntimeState field views;
- wrong tensor shape, dtype, device, contiguity, or finite status;
- a denominator different from `1 - dt * L`;
- mismatched projector/backend, pointwise-kernel, or Q-gradient-cache
  identity;
- incompatible physical scalar coefficients or execution policy;
- a changed projected-Euler operation graph;
- an already evolved state or active workspace;
- a nonzero legacy workspace, which would require a shared private assembly
  factory before proceeding.

## Workspace decision

The current qualified legacy assembly retains a zero-slot, zero-byte
`RuntimeWorkspace`.  The binding adds zero workspace bytes and introduces no
new storage identity.  Therefore P5.2 does not need to split the assembly
factory.  This decision is local to the observed qualified construction; the
gate will fail if the legacy workspace later becomes nonempty.

## Deliberately unchanged

P5.2 does not change:

- `PlaneRuntimePath`, CLI choices, or the production default;
- either existing Plane runtime or its import graph;
- equations, kernels, transforms, boundaries, operation order, state,
  checkpoint format, outputs, or metadata;
- the Phase 6 authorization or default-promotion state.

## Local qualification

The current local qualification completed with:

- 173 targeted tests passed with no failure, skip, or deselection;
- 1883 full-suite tests and 8 subtests passed with no failure, skip, or
  deselection;
- `git diff --check` passed;
- the architecture import boundary remained closed, with this private binder
  added only to the exact Phase 2 decomposition-consumer allowlist.

P5.3 integration additionally corrected the operator references from direct
model calls to the exact integrator `_prepare_algebraic` and `_explicit_rhs`
operations.  This does not change the numerical kernels; it prevents the
compiled consumer from bypassing the qualified restored-static-field
invalidation rule.  The hashes in the machine-readable record describe this
corrected binding and the complete suite was rerun afterward.

## Next boundary

P5.3 is locally eligible but not yet implemented.  It may create a private
compiled projected-Euler step program over these pre-bound references and a
fully preallocated bounded workspace.  It must freeze callback placement,
projection, inverse transforms, spectral refresh, failure atomicity, and
progress commit, then compare one-step and short continuous CPU results with
the legacy oracle.  It must remain disconnected from the application edge.

The machine-readable authority is
`phase_5_p52_construction_binding.json`.
