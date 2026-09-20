# Phase 3 P3.4 H100 qualification

Status: `COMPLETE_WITH_AUTHORIZED_RECOVERY_V3`.

Final classification:
`PASS_P3_4_SEPARATED_CANARY_NON_REGRESSION_WITH_AUTHORIZED_RECOVERY_V3`.

Qualified implementation: `4406868087b3b935bb194b8026a3bf1c0fd19ab6`.
Frozen parent: `b425ba9488253cda5f7b5ebe6d0fdf1a35f5eab8`.

The explicit `separated_canary` connection of `RuntimeState`, bounded
workspace, and `StepProgram` passed its independent CPU and H100 gates.  The
production default remained `legacy_production`; this result authorizes P3.5
work but does not promote a runtime or change a public API.

## Numerical and restart evidence

- R128 parent/candidate initialization and final Q/u/p were byte-for-byte
  identical and matched the frozen Phase 2 canary oracle.
- R320 parent/candidate initialization and final Q/u/p were byte-for-byte
  identical.
- Parent-step-50 to candidate-step-100 and candidate-step-50 to
  parent-step-100 restarts matched the continuous R128 oracle byte-for-byte.
- Checkpoint tensor tampering, metadata/hash tampering, runtime-identity
  mismatch, and cross-runtime restart were rejected before a scientific
  timestep or formal output-directory creation.
- Checkpoint format remained version 1.

The frozen R128 canary hashes are:

| Output | SHA-256 |
|---|---|
| `Q_100.npy` | `f2a94bf9c55e71632f5397b67ff1dfdf34a934981dd27514ab3c8a5546c738f2` |
| `u_100.npy` | `25ebb2f4ec51156d0504624a88ab336a9baaee58efa7f2d7f01d6a2fc0e65b33` |
| `p_100.npy` | `44b9d577bb328537f3a67f876fbe1d34f390b7821761f2ed4f8f8b7db3b82a37` |

The R320 parent/candidate hashes are:

| Output | SHA-256 |
|---|---|
| `Q_100.npy` | `95765115b54173bc25ae091ea45135f696a336b266a01d69c076290cfafefe51` |
| `u_100.npy` | `9463a8cf395dd2e5b6c9a7b5539611949542594a60293f0d48b590c661d7b12b` |
| `p_100.npy` | `9ea17a1a6f69f66ae657c13ebf4af1d55f68248795665de3e17148500e965e47` |

## Performance evidence

All twelve H100 profiler records completed with seven forward and 32 inverse
transforms per timestep, no graph breaks or compile/projected-transform
fallbacks, and no OOM, NaN, Inf, or CUDA error.

| Grid | Parent mean (ms) | Candidate mean (ms) | Mean B/A | Median B/A | Allocated/reserved B/A |
|---|---:|---:|---:|---:|---:|
| R128 | 5.854829 | 5.792985 | 0.989437226 | 0.991014164 | 1.0 / 1.0 |
| R320 | 45.232064 | 45.221354 | 0.999763203 | 1.000719778 | 1.0 / 1.0 |

## Evidence chain

The qualification used one primary job and three explicitly authorized
recovery jobs.  The first three jobs retained valid partial evidence but
stopped because of external qualification-harness defects: a construction
metadata snapshot was interpreted as final state, a helper namespace was
mistyped, and a pressure-oracle literal was truncated.  None was a solver or
scientific failure.  Recovery v3 completed the remaining gates and preserved
all earlier failed states in the final evidence index.

| Stage | Job | State | Evidence manifest |
|---|---:|---|---|
| Primary | 10835768 | FAILED (external metadata validator) | `53ba5c67fd2250bddfa2a9aee8b26b20a2f724dcf81397ae97aa75090d7841fd` |
| Recovery v1 | 10835784 | FAILED (external driver namespace) | `28026c55175fc51d90615a24d078c053a497b8870767e4f81fd00169566acc52` |
| Recovery v2 | 10835787 | FAILED (external oracle literal) | `f514c9415158026f2d9a0d9f3a3d463dedac014fa30209ca38723b51b2e18687` |
| Recovery v3 | 10835801 | COMPLETED | `2e23ccbd820894559ecd65b6fe70c1baa93dbad7ed4cf6244c4b19233b466307` |

The authoritative completed evidence directory is
`/home/fansenwei/pssolver_phase3_p34_recovery_4406868_20260920_v3`.
Its checksum manifest passed 134/134 entries and its `COMPLETE` marker is
present.

## Decision

P3.4 is closed and P3.5 is authorized.  P3.5 must preserve the public
production facade, default runtime selection, checkpoint format v1, metadata
and output schedules, operation order, numerical oracles, transform counts,
and performance/memory gates.  P3.6 remains a separate closure qualification.
