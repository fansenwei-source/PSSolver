# Architecture Stage O.4: bounded Plane production-canary qualification

## Scope

Stage O.4 supplies the evidence pipeline for comparing the unchanged
`legacy_production` Plane runtime with the explicit `separated_canary` through
the shared Stage O.3 workflow.  It does not change the omitted runtime default,
authorize promotion, migrate Channel, alter the generic solver, or run a long
benchmark.

The local implementation adds read-only workflow comparison and qualification
analysis, a planning-only command generator, and CPU characterization tests.
The actual performance and memory decision requires one separately authorized
H100 job.

## Frozen numerical and restart gates

Both runtime paths use the same `128 x 128 x 32`, float64, TF32-off Plane
configuration, production numerical policies, seed, and initial-condition
construction.  The H100 plan generates:

- a six-step run saving every Q/u/p frame on both runtime paths;
- a 100-step run saving Q/u/p at steps 0 and 100 on both paths;
- a three-step checkpoint plus three-step continuation on each path.

The cross-runtime comparisons require an exact initial-Q SHA-256 match and a
maximum relative L2 error of `1e-10`.  Q and velocity use raw relative L2;
pressure records both raw and demeaned errors and gates on the demeaned value.
Every array must have the same shape, float64 dtype, and finite values.  Both
same-backend restart comparisons must be byte-exact at step 6.  Scientific
metadata signatures, output frame sets, complete metadata, and the final-write
`COMPLETE` contract are validated before numerical acceptance.

## H100 performance gate

Three balanced legacy/canary pairs use ten warm-up and twenty timed steps.  The
legacy profiler's O.4-only `whole_timestep` timing scope records only the outer
CUDA event pair while continuing to count nested transform calls.  This avoids
bias from nested profiler events and matches the canary's outer-step timing
method without changing the profiler's established default behavior.

The fixed non-regression limits are:

- aggregate canary/legacy mean timestep ratio at most `1.03`;
- at least two of three paired ratios at most `1.05`;
- peak allocated and reserved memory ratios at most `1.03`;
- identical forward and inverse transform calls per step;
- identical, valid canary algebraic-lifecycle counters across all trials;
- no retained generation tensors or on-demand physical materialization.

An accepted report is `A_recommended` and is eligible only for a separate
Stage O.5 decision.  It explicitly keeps
`eligible_for_production_promotion=false`.

## Safety and auditability

The planner requires nonexistent, independent control and scratch roots and
only prints argument vectors.  HPCC execution must use one clean detached
worktree at the exact O.4 commit, one H100 job, no requeue, and no automatic
retry.  Any failed command, provenance mismatch, incomplete output, non-finite
array, numerical/restart failure, or performance/memory/lifecycle gate failure
stops the stage without changing defaults or existing data.

The benchmark `develop` workspace is outside this stage and remains untouched.

## Local characterization

The real CPU workflow was exercised independently of the synthetic analyzer
tests.  On an `8 x 8 x 8` float64 case, the legacy/canary maximum relative L2
error was `9.81e-17` over all 21 Q/u/p arrays in the six-step trajectory and
`8.01e-16` over the step-0/step-100 arrays.  Initial Q was exact.  Both legacy
and canary three-plus-three-step restarts were byte-exact against their
continuous six-step trajectories.  These results qualify the local pipeline;
they do not substitute for the frozen H100 performance and memory gates.
