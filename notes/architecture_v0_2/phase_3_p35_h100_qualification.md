# Phase 3 P3.5 H100 qualification

Status: `COMPLETE_WITH_VALIDATOR_RECOVERY_V2`.

Final classification:
`PASS_P3_5_PRODUCTION_FACADE_CONNECTION_NON_REGRESSION_WITH_VALIDATOR_RECOVERY_V2`.

Qualified implementation: `5bb796fa75a6d30afc37251f86dbd1b1723efa51`.
Frozen parent: `54cffec80ef8aab63988f8cd65dc66600b48452c`.

The state-backed integrator core connected beneath the stable
`legacy_production` facade passed its independent CPU and H100 qualification.
The public facade, metadata surface, checkpoint format, production default,
and numerical results remained unchanged.  `separated_canary` remains an
explicit non-default path; this result authorizes P3.6 closure qualification
but does not promote either runtime or change a public API.

## Numerical evidence

- R128 legacy parent/candidate initialization and final Q/u/p were
  byte-for-byte identical and matched the frozen production oracle.
- R128 separated-canary parent/candidate initialization and final Q/u/p were
  byte-for-byte identical and matched the frozen canary oracle.
- R320 legacy parent/candidate initialization and final Q/u/p were
  byte-for-byte identical.
- Parent-step-50 to candidate-step-100 and candidate-step-50 to
  parent-step-100 legacy restarts matched their continuous targets
  byte-for-byte.
- Reset/rebind passed all twelve GPU gates, including zero-copy storage,
  device and dtype preservation, progress preservation, zero-byte workspace,
  and a finite subsequent step.
- Tensor tampering, metadata/hash tampering, runtime-identity mismatch, and
  cross-runtime restart were rejected as required.

The frozen R128 legacy hashes are:

| Output | SHA-256 |
|---|---|
| `Q_100.npy` | `d20547158acba4d8ab6b9e3c6c0cc48954c6b4b6ce9af229fe826d510e577796` |
| `u_100.npy` | `1199675fe7c1a0dcc4ec1f6a0c9130021a30be94e33dab523325c22076ec5a64` |
| `p_100.npy` | `34ef4007db9f900f1902b5f3cd43efe83a82cbbc393dd8d0df6229613fc848ea` |

The frozen R128 separated-canary hashes are:

| Output | SHA-256 |
|---|---|
| `Q_100.npy` | `f2a94bf9c55e71632f5397b67ff1dfdf34a934981dd27514ab3c8a5546c738f2` |
| `u_100.npy` | `25ebb2f4ec51156d0504624a88ab336a9baaee58efa7f2d7f01d6a2fc0e65b33` |
| `p_100.npy` | `44b9d577bb328537f3a67f876fbe1d34f390b7821761f2ed4f8f8b7db3b82a37` |

The R320 legacy parent/candidate hashes are:

| Output | SHA-256 |
|---|---|
| `Q_100.npy` | `d30ac02728ac1d3e52b4fd8ad7385c4bf747e7e4615ba454e929b2c2538041eb` |
| `u_100.npy` | `f2f169baa4b78441bcc83abd8063420055759f61aa0828e69c120a0412681e82` |
| `p_100.npy` | `118dd56bf56eaf54cea3fe2b3bf75487975077e7c985f5356d395e5e5c8a864a` |

## Performance evidence

All twelve H100 profiler records completed with seven forward and 32 inverse
transforms per timestep, no graph breaks or compile/projected-transform
fallbacks, and no OOM, NaN, Inf, or CUDA error.

| Grid | Parent mean (ms) | Candidate mean (ms) | Mean B/A | Median B/A | Allocated/reserved B/A |
|---|---:|---:|---:|---:|---:|
| R128 | 5.730392 | 5.782796 | 1.009144895 | 1.017766964 | 1.0 / 1.0 |
| R320 | 45.097170 | 45.020380 | 0.998297218 | 0.997492328 | 1.0 / 1.0 |

## Identity and metadata contract

The legacy adapter intentionally records
`runtime_selection.separated_architecture = null`.  Its P3.5 internal state,
workspace, and step-program identity is verified by an independent in-process
GPU connection smoke rather than by reading canary-only metadata through the
legacy trajectory document.  That smoke passed all 14 gates.

Schema-v1 `canonical_sha256()` includes the complete run request, including
`output_dir`.  Parent and candidate trajectory hashes therefore differ by
design when their output directories differ.  Each saved canonical hash was
recomputed against its own resolved request and matched.  Complete resolved
request comparisons found only `$.output_dir`; runtime-identity hashes were
equal for each parent/candidate pair.  This preserves both provenance and the
numerical/restart identity contract without conflating them.

## Evidence chain

The first job failed before its script body because the external harness used
a stale Slurm log path.  The first recovery produced valid profiler and R128
legacy evidence, then stopped because its external validator applied canary
metadata and canonical-identity rules to legacy trajectories.  Recovery v2
corrected only that external validator, reused 134 rehashed evidence files,
and completed all remaining scientific gates.  Neither earlier failure was a
solver or scientific failure.

| Stage | Job | State | Scientific result |
|---|---:|---|---|
| Primary | 10835804 | FAILED before script body | no scientific execution |
| Recovery v1 | 10835806 | FAILED in external validator | valid partial evidence |
| Recovery v2 | 10835824 | COMPLETED | all P3.5 gates passed |

The authoritative completed evidence directory is
`/home/fansenwei/pssolver_phase3_p35_h100_5bb796f_20260920_recovery_v2`.
Its stable checksum manifest passed 727/727 entries, has SHA-256
`59f92f35f5f8ad36b0a0dc81954d3bce01049e5c4f4b82b507fe09c4b8d8ddf4`,
and its `COMPLETE` marker is present.

## Decision

P3.5 is closed and P3.6 is authorized.  P3.6 remains an independent closure
qualification covering the complete CPU and package suites, checkpoint-v1 and
tamper contracts, continuous/split trajectories, and balanced R128/R320 H100
performance, memory, transform-call, fallback, and finite-value gates.  A
runtime promotion or production-default change still requires a separate,
explicit decision.
