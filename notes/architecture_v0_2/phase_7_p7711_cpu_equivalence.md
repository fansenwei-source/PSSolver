# Phase 7 P7.7.11: public-entry CPU identity and restart qualification

Status: `P7_7_11_COMPLETE_P7_7_12_NOT_STARTED`.

Parent public-runner baseline:
`6bb269f` on `next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_11_CPU_BYTE_IDENTITY_AND_RESTART`.

P7.7.11 qualifies the P7.7.10 public execution connection against the
pre-existing Plane and Channel application entries. The machine-readable
authority is
[phase_7_p7711_cpu_equivalence.json](phase_7_p7711_cpu_equivalence.json).

## Frozen CPU matrix

The qualification covers every application/runtime combination currently
registered by the public compiler:

| application | runtime |
|---|---|
| Plane complete-stress Beris--Edwards | `legacy_production` |
| Plane complete-stress Beris--Edwards | `compiled_v2` |
| Channel active-force active nematics | `legacy_channel` |
| Channel active-force active nematics | `compiled_channel_v2` |

Plane uses float64 on an `8 x 8 x 6` slab. Channel uses its qualified float32
contract on an `8 x 6 x 5` rectangular domain. Both use eager pointwise CPU
execution and two deterministic timesteps. These are connection oracles, not
physical-resolution benchmark runs.

For each matrix cell the test executes the same declaration through:

1. `compile_simulation()` followed by the existing application runner;
2. `compile_simulation()` followed by public `run_simulation()`.

All saved Q/u/p NPY files, diagnostics NPY and CSV files, and COMPLETE markers
are byte-for-byte identical. Workflow steps, runtime selection, numerical
metadata, boundary metadata, final observation arrays, and diagnostic records
also agree. The final Q/u/p and diagnostics file SHA-256 values are frozen in
the machine-readable authority rather than being recomputed into the record.

## Same-runtime restart matrix

Each public entry first produces a one-step segment and checkpoint. The same
checkpoint is then consumed independently by:

- the existing application runner; and
- public `run_simulation()`.

Both resumed executions start at step 1 and finish at step 2. Their complete
output artifacts are byte-for-byte identical, and their final Q/u/p files are
also identical to the corresponding uninterrupted two-step public trajectory.
The metadata records the same source runtime path and source completed-step
clock. No cross-runtime restart or fallback is allowed or claimed.

This establishes that the public declaration, compiler, runner registry, and
`SimulationResult` wrapper do not alter the existing checkpoint/restart
semantics.

## Full-complex Plane identity correction

The first CPU qualification attempt exposed a metadata inconsistency in the
existing Plane assembly. A valid public `SpectralNumerics` request using
`full_complex` storage declares `hermitian_axis=None`, because full-complex
storage has no truncated Fourier axis. The Plane runtime construction
nevertheless passed the geometry's periodic axis `1` to the transform backend.

The legacy path could run because that value is numerically unused for
full-complex storage, but the strict `compiled_v2` binding correctly rejected
the mismatch between the declared and allocated layouts. The construction now
passes the already validated `numerics.hermitian_axis` instead.

Consequences of the correction:

- full-complex construction now has `hermitian_axis=None` consistently;
- production and benchmark `hermitian_half` construction remains axis `1`;
- no transform algorithm, tensor value, timestep operation, coefficient,
  boundary condition, or runtime default changes;
- the corrected full-complex legacy and compiled paths produce identical
  frozen Q/u/p and diagnostic hashes.

## Scope and non-claims

P7.7.11 changes no public declaration or result field. It does not promote a
compiled runtime, change Plane or Channel defaults, add a model/geometry
combination, authorize a boundary condition, or modify an on-disk schema.

The test intentionally proves CPU identity and same-runtime restart only. It
does not qualify GPU performance, GPU memory, CUDA device binding, production
grid sizes, or long-time scientific behavior. Those remain the responsibility
of P7.7.12 and existing benchmark evidence.

## Verification

The exact targeted and complete CPU test counts are recorded in the
machine-readable authority after final regression. The frozen test validates:

- four direct/public continuous-run comparisons;
- four public-checkpoint/direct-resume comparisons;
- four public-checkpoint/public-resume comparisons;
- four uninterrupted/resumed final-state comparisons;
- Q/u/p, diagnostics, workflow metadata, runtime identity, and public result
  agreement;
- the reviewed P7.7.10 source identities plus the Plane construction fix.

## Next boundary

P7.7.12 is the next and final P7.7 task: one fail-closed H100 non-regression
closure must exercise both application combinations and both runtime paths,
verify numerical identity and restart on CUDA, and check that the public layer
adds no meaningful timestep or memory regression. Phase 8 and Phase 9 remain
unauthorized until that closure passes.
