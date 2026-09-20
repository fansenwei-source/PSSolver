# Phase 2 qualification

Status: `COMPLETE_H100_PASS_WITH_AUTHORIZED_RECOVERY`

Final classification:
`PASS_PHASE2_NON_REGRESSION_WITH_AUTHORIZED_RECOVERY`.

- v0.1.2 baseline: `4fa614616e9d67c98be52c150eaf303a0c80b5c1`;
- Phase 2 qualification commit: `6f054f913ce3bd22eb095e11ad47ca3736cfc76d`;
- qualified implementation: `4260480989169e817fba4af1949c15c44223b248`;
- initial H100 Job: `10835683`;
- authorized recovery Job: `10835702`;
- machine-readable evidence:
  [phase_2_h100_qualification.json](phase_2_h100_qualification.json).

## Architecture outcome

Phase 2 decomposed the legacy Plane Beris--Edwards RunSpec behind its frozen
compatibility facade.  Tensor-free model, system, geometry, numerical,
workflow, and invocation declarations now have explicit ownership.  The
schema-v1 serializer remains the compatibility authority, and the legacy
runtime selector, workflow, application, and configuration consumers use the
component graph without changing their public contract.

The two exact configuration reverse dependencies were retired.  Production
implementation provenance now covers all 54 sources that determine the Plane
application, including every imported PSSolver package facade.  Four unrelated
legacy architecture exceptions remain explicitly ratcheted for later phases.

Phase 2 did not change equations, boundary conditions, transform algorithms,
timestep order, floating-point operation order, output schema, runtime default,
or any qualified production default.  `legacy_production` remains the default;
`separated_canary` remains an explicit non-promoted canary.

## CPU and compatibility evidence

The HPCC candidate gate passed 364 targeted tests.  The complete baseline and
candidate suites passed 1131 and 1620 tests respectively, with 8 subtests; the
only skip was the existing optional Nematics3D dependency and the two
CUDA-only tests were run successfully on the H100.

Local qualification had already established byte-identical continuous output
and bidirectional cross-version restart for both runtime paths, equal
schema-v1 metadata outside implementation provenance, and clean wheel/sdist
installation.  The final H100 task independently exercised the production
scale trajectory and performance gates.

## H100 non-regression

All 12 balanced profiler records passed.  Transform calls remained `7/32` per
step, with no graph breaks, compile fallback, projected-transform fallback,
OOM, NaN, Inf, or CUDA error.

| Grid | Baseline mean (ms) | Candidate mean (ms) | Mean ratio | Median ratio | Allocated/reserved ratio |
|---|---:|---:|---:|---:|---:|
| R128 | 5.717263 | 5.739138 | 1.003826 | 1.003792 | 1.0 / 1.0 |
| R320 | 45.262218 | 45.236063 | 0.999422 | 0.999773 | 1.0 / 1.0 |

Both grids are comfortably inside the frozen 1.02 non-regression limits.

Four 100-step trajectories covered baseline/candidate and
`legacy_production`/`separated_canary`.  Same-runtime baseline and candidate
`Q/u/p` were byte-identical.  The legacy outputs exactly matched the frozen
v0.1.2 oracle.  Both runtime paths resumed from step 50 to step 100 with final
`Q/u/p` byte-identical to continuous execution, while a cross-runtime restart
was rejected before output creation or checkpoint tensor loading.

## Authorized recovery

Job `10835683` completed every performance and continuous-trajectory gate but
was marked failed by an external validation helper that incorrectly required
restart `saved_steps=[100]`.  The workflow's absolute schedule legally records
`[50,100]`: step 50 satisfies the save schedule and step 100 is the mandatory
final observation.  This was a validation-contract error, not a solver or
scientific failure.

The original 119-file manifest remained unchanged and passed before and after
recovery.  The single authorized recovery Job `10835702` validated the correct
schedule with seven synthetic tests, revalidated the existing legacy restart,
completed only the missing canary restart and cross-runtime rejection gate,
and produced a separate passing 36-file manifest.  It did not rerun the
profilers, continuous trajectories, CPU suites, CUDA tests, or legacy restart.

## Decision

Phase 2 is complete.  This qualification does not promote a runtime, change a
default, merge the architecture branch into `develop`, or automatically
authorize Phase 3.  The next architecture phase must begin with its own frozen
design and compatibility contract.
