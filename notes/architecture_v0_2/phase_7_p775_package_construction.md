# Phase 7 P7.7.5: package construction ownership normalization

Status: `P7_7_5_COMPLETE_P7_7_6_NOT_AUTHORIZED`.

Parent baseline: `0fed57f04db1b91e67cebcac577ac63bd2427318` on
`next/pssolver-v0.2.0-architecture`.

Classification:
`PASS_P7_7_5_PACKAGE_CONSTRUCTION_OWNERSHIP_NORMALIZATION`.

P7.7.5 normalizes the remaining construction ownership identified by P7.7.4.
All four qualified Plane and Channel runtime paths can now be constructed from
one frozen package plan and one existing typed request, without an
application-supplied builder closure.

The machine-readable authority is
[phase_7_p775_package_construction.json](phase_7_p775_package_construction.json).

## Historical layering is preserved

P7.7.4 remains unchanged and continues to describe the ownership that existed
at its own commit. P7.7.5 adds a new `PackageRuntimeConstructionPlan` over the
P7.7.4 binding rather than rewriting that historical record.

The new plan binds:

- the P7.7.4 binding SHA-256;
- source `SimulationSpec` SHA-256;
- `SimulationLoweringPlan` SHA-256;
- exact runtime kind and typed request class;
- central package factory;
- package-owned implementation builder;
- adapter protocol and no-fallback policy.

It contains no callable, tensor, runtime object, registry, or mutable state.

## Typed construction input

`PackageRuntimeConstructionInput` contains exactly two fields:

1. a frozen `PackageRuntimeConstructionPlan`;
2. an existing `PlaneRuntimeBuildRequest` or `ChannelRuntimeBuildRequest`.

It has no builder field. Before allocation, the package factory recomposes the
simulation declaration from the request, regenerates the expected package
plan, and requires exact equality. A plan from a different runtime path,
geometry, grid, boundary assignment, numerical option, or initial-condition
declaration is rejected before any builder runs.

## Package-owned builders

The four paths resolve to:

- Plane legacy: the existing package-owned `build_legacy_plane_runtime`;
- Plane compiled: a lazy typed bridge to the existing compiled workflow
  builder;
- Channel legacy: a new typed package builder containing the exact former
  application assembly;
- Channel compiled: the existing typed bridge, now imported lazily.

The bridges validate the request type and selected runtime path before calling
their existing implementations. They do not catch construction failures or
fall back to another runtime.

## Compatibility debt is explicit

This slice does not pretend that old module placement is already ideal. Three
exact dependency edges remain named migration debts:

- Plane runtime bridge to the historical compiled workflow module;
- Channel runtime bridge to the historical `pssolver.channel` implementation;
- Channel runtime bridge to the experimental compiled implementation.

All three imports are lazy, so selecting another runtime does not import the
unused implementation. Broad runtime-to-workflow, runtime-to-experimental, or
runtime-to-legacy-module dependencies are not allowed; only these exact edges
are recorded by the architecture ratchet.

Future physical relocation can retire these edges independently after object,
checkpoint, numerical, and performance qualification. P7.7.5 changes
ownership first and avoids mixing it with a large code move.

## Numerical and object equivalence

The normalized package path and the prior direct construction path are run
independently for two CPU steps for all four targets. Physical fields,
spectral fields, linear state where applicable, and progress are byte-for-byte
identical.

A sentinel test also proves that the central package factory returns the exact
adapter object produced by the P7.7.4 bound factory. No adapter wrapper or
tensor copy is added.

## Performance boundary

Plan generation and identity validation run once before allocation. Once a
runtime exists, its solver, transforms, operators, state, workspace, and
`advance()` method are unchanged. There is no construction registry or
capability lookup in the timestep.

Therefore P7.7.5 adds no steady-state timestep work and no persistent device
allocation. CPU equivalence is sufficient for this still-disconnected opt-in
slice. H100 qualification remains mandatory when a later slice changes the
production application construction path.

## Production and public API status

Neither production application imports the new package factory. The Plane
application still uses its previous construction code, and the Channel
application still uses its previous construction code. Defaults remain:

- Plane: `legacy_production`;
- Channel: `legacy_channel`.

The new plan, input, and factory are not exported from package roots. No CLI,
metadata, output, checkpoint, pickle, or public API contract changes.

## Verification

The final local results are:

```text
targeted: 204 passed
complete: 2118 passed, 8 subtests passed
```

## Next slice

P7.7.6 is not authorized by this result. A separately authorized slice may
replace the application-local builder closures with
`PackageRuntimeConstructionInput` and `build_package_simulation_runtime`.
That production connection must pass complete CPU regression, short
byte-identity trajectories, checkpoint/restart gates, and H100 performance and
memory non-regression before any default promotion.

P7.7.5 does not relocate numerical kernels, add boundary physics, promote a
compiled runtime, authorize Phase 8, or authorize Phase 9.
