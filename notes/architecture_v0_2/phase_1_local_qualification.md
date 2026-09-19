# Phase 1 qualification

Status: `COMPLETE_H100_PASS`

Final classification: `PASS_PHASE1_NON_REGRESSION`

- v0.1.2 baseline: `4fa614616e9d67c98be52c150eaf303a0c80b5c1`;
- qualified candidate: `276f14ea08d372a384f87ce39ffe91f2b4f6fbc1`;
- H100 qualification Job: `10835314`;
- machine-readable evidence:
  [phase_1_h100_qualification.json](phase_1_h100_qualification.json).

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
8. `0f9a6bf` — adopt canonical imports where allowed by the dependency graph;
9. `db6964d` — record local qualification;
10. `276f14e` — satisfy the frozen whitespace/static gate.

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

This control was supplementary to the frozen v0.1.2 H100 oracle.  The formal
R128/R320 phase-completion task described below subsequently passed.

## Formal H100 phase-completion result

The balanced H100 task compared the release baseline with the final candidate
in one allocation on an NVIDIA H100 PCIe.  Job `10835314` completed with exit
code `0:0`; both real-GPU CUDA-only tests passed, and all twelve profiler JSON
records were complete.

| Grid | Baseline mean (ms) | Candidate mean (ms) | Candidate/baseline |
|---|---:|---:|---:|
| R128 (`128 x 128 x 32`) | 5.653715 | 5.636842 | 0.997016 |
| R320 (`320 x 320 x 80`) | 45.143812 | 45.137972 | 0.999871 |

Peak allocated and reserved memory ratios were exactly `1.0` at both grids.
Forward/inverse transform counts remained `7/32`; graph breaks, compile
fallbacks, and projected-transform fallbacks were all zero.  There was no OOM,
NaN, Inf, or CUDA error.

The baseline and candidate 100-step production trajectories were
byte-identical and exactly reproduced the frozen v0.1.2 oracle:

| Array | SHA-256 |
|---|---|
| `Q_100.npy` | `d20547158acba4d8ab6b9e3c6c0cc48954c6b4b6ce9af229fe826d510e577796` |
| `u_100.npy` | `1199675fe7c1a0dcc4ec1f6a0c9130021a30be94e33dab523325c22076ec5a64` |
| `p_100.npy` | `34ef4007db9f900f1902b5f3cd43efe83a82cbbc393dd8d0df6229613fc848ea` |

A pickle-free checkpoint written by v0.1.2 at step 50 was resumed by the
candidate to step 100.  Its Q/u/p outputs were byte-identical to both continuous
runs and the frozen oracle.  This closes the checkpoint and module-provenance
risk created by moving the concrete classes.

The authoritative HPCC control directory is:

```text
/home/fansenwei/pssolver_phase1_transform_extraction_h100_ab_276f14e_20260918_v2
```

Its 120-entry checksum manifest passed completely.  The manifest's own
SHA-256 is
`26120611f8e0f38347b35566079f82350b67d624b07ccb31563ed10c68afa630`.

All gates in
[phase_1_transform_extraction.md](phase_1_transform_extraction.md) passed.
Phase 1 is complete.  This result does not change a production default,
authorize a merge to `develop`, remove the compatibility facade, or promote a
new runtime because only module ownership changed.
