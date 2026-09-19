# Phase 1 local qualification

Status: `LOCAL_PASS_H100_PENDING`

Phase 1 mechanically extracted the v0.1.2 transform responsibility cluster.
It did not change equations, basis definitions, transform order, allocation
policy, runtime selection, Plane defaults, or output schemas.

## Canonical ownership after extraction

| Responsibility | Canonical module |
|---|---|
| Tensor-product transform backend | `pssolver.backends.tensor_product` |
| Spectral projection | `pssolver.operators.projection` |
| Qualified 3D tensor divergence | `pssolver.operators.tensor_divergence` |
| Plane free-slip Stokes solve | `pssolver.linear_solvers.stokes.plane_free_slip` |
| Historical import compatibility | `pssolver.transforms` |

`pssolver.transforms` contains no numerical class or function definitions.  It
directly aliases the 18 frozen v0.1.2 compatibility names and has an explicit
`__all__`.  There are no wrappers, subclasses, duplicated constants, or
fallback implementations.

## Reviewable implementation sequence

1. `dff450e` — characterize the transform compatibility facade;
2. `dcf44de` — create canonical numerical module packages;
3. `29403d1` — extract the tensor-product transform backend;
4. `908efa8` — extract spectral projection operators;
5. `9580014` — extract projected tensor-divergence operators;
6. `b7aac0f` — extract the Plane free-slip Stokes solver;
7. `2748590` — finalize the compatibility facade;
8. `0f9a6bf` — adopt canonical imports where allowed by the dependency graph.

For each extracted implementation block, an automated source comparison
confirmed that the definition body was byte-preserved.  Only module imports,
ownership, provenance inventory, and tests changed around those blocks.

## Compatibility evidence

The executable facade contract freezes:

- all 18 direct `pssolver.transforms` names;
- all 14 root `pssolver` transform re-exports;
- constructor, function, and public-method signatures;
- constants and shared-object identity;
- canonical `__module__` provenance;
- legacy protocol-0 pickle global resolution through the facade;
- the free-slip Stokes state-dict and registered-buffer schema;
- pickle-free supported Plane snapshots and checkpoints.

Configuration and `models.active_nematics.stokes` retain their two intentional
facade dependencies.  Moving them directly to execution modules would violate
the accepted target layering; those debts require later specification/model
decomposition rather than an import-only rewrite.  The Plane runtime's former
facade dependency was safely retired.

## Local numerical identity control

A real three-step `legacy_production` Plane run was executed before and after
the extraction with CPU float64, eager pointwise execution, an `8 x 8 x 8`
grid, activity number 18, hydrodynamic output enabled, and the same seed and
defaults.  The final arrays were byte-identical:

The machine-readable record is
[phase_1_local_identity_control.json](phase_1_local_identity_control.json).
The baseline executed at `dff450ed4d2fe9d293b0a05f7b8e7f36aca5fd9e`,
whose numerical source tree still matched release `v0.1.2`; the candidate
executed at `0f9a6bf6571519c2f5b436b4c44ed4594fb93c09`.

| Array | SHA-256 |
|---|---|
| `Q_3.npy` | `a935874bfd01ad89cdb23ae7dd450c1c9654eb469f826958e5f3bf48910f5f50` |
| `u_3.npy` | `3e6544aa24f563245deffdd9d2fc87c8ac9816874a99b97dfad6da467fe6f7f5` |
| `p_3.npy` | `22bc104a31b051e068be5cefd3d4866e43a8de5439bbcc943f6d4bd009bce07b` |

This control is supplementary to the frozen v0.1.2 H100 oracle; it is not a
replacement for the formal R128/R320 phase-completion task.

## Remaining gate

Local qualification does not authorize integration or default promotion.  One
balanced H100 task must still compare v0.1.2 with the final Phase 1 candidate
at R128 and R320, including three paired trials, transform-call counts, peak
memory, a 100-step byte-identity trajectory, restart, and output-schema checks.
The thresholds and stop conditions remain those in
[phase_1_transform_extraction.md](phase_1_transform_extraction.md).
