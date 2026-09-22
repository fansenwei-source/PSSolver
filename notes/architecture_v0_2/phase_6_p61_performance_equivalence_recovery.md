# Phase 6 P6.1: performance-equivalence recovery

Classification: `READY_PHASE6_PERFORMANCE_EQUIVALENCE_RECOVERY`.

Profile source commit:
`bad9916ac1cae4969372712edf88d106d529d584`.

Original H100 Job: `10837343` on `gpu-h100-4-0`.

Original evidence manifest SHA-256:
`ba00f7f5bc64cd5ae022d53d6eab3505636b0384fd047b8650e91b7d3167133c`.

The original frozen contract classified the run as
`FAIL_PHASE6_H100_CANDIDATE_GATE` because `compiled_v2` was faster in only one
of three paired trials at each grid.  That classification remains immutable
and is not relabelled by this record.  The Job completed all twelve profilers
and was then intentionally cancelled before completing the workflow matrix.

## Evidence from the failed gate

All profiler identity, numerical, transform-call, graph-break, finite-state,
mean-time, median-time, and memory gates passed.  Paired initial-Q and final
state SHA-256 identities matched.  The only failed predicate was the
directional paired-win count.

At R128, the candidate/baseline mean ratio was `0.998114208`, the median ratio
was `1.000087885`, and the three paired ratios were `1.000735697`,
`1.007441320`, and `0.986209464`.  At R320, the mean ratio was `1.000100141`,
the median ratio was `1.000225270`, and the paired ratios were `1.000225270`,
`0.999792755`, and `1.000282881`.  Peak allocated and reserved ratios were at
most `1.002325581`.

The sign of a sub-percent timing difference is not a stable qualification
predicate for an architecture refactor whose requirement is performance
non-regression.  In particular, the R320 sign changes around unity at the
fourth decimal place.  Requiring two strict wins out of three can reject two
statistically indistinguishable implementations solely because timing noise
falls on different sides of one.

## Corrected performance contract

The paired-win count remains recorded but becomes informational.  Each grid
must instead satisfy every one of the following fail-closed gates:

- candidate/baseline mean timestep ratio at most `1.02`;
- candidate/baseline median timestep ratio at most `1.02`;
- every individual paired timestep ratio at most `1.02`;
- peak allocated-memory ratio at most `1.02`;
- peak reserved-memory ratio at most `1.02`.

A grid is `accelerated` only when mean and median ratios are both at most
`0.98` and at least two paired trials are faster.  A grid that passes all
non-regression gates without that material speedup is
`performance_equivalent`.  Any non-regression failure is `regressed`.

This correction does not alter an equation, numerical operation, threshold
for regression, runtime implementation, profiler artifact, or production
default.  It replaces only an unstable directional-benefit requirement with
an explicit equivalence classification.  Performance-only adjudication never
claims long-run eligibility.

## Recovery provenance and scope

The analyzer binds the immutable profiler evidence to the original source
commit and binds new workflow evidence to a separate qualification commit.
An authorized recovery must also bind the original manifest SHA-256, record
the cancelled source Job ID `10837343` with twelve complete profilers, record two total
formal H100 submissions, and forbid automatic retry.

The recovery may reuse only the twelve profiler JSON files after the entire
original `1670/1670` manifest is revalidated.  It must use a new control and
scratch root.  Its single H100 Job may run only the missing production
trajectory, restart, negative, diagnostic, metadata, and auxiliary gates.  It
must not rerun the profilers or start T=400.

Only a complete recovery ending in
`PASS_PHASE6_H100_CANDIDATE_PERFORMANCE_EQUIVALENT` or
`PASS_PHASE6_H100_CANDIDATE_ACCELERATED` can make the separately authorized
long-run stage eligible.  Neither result promotes `compiled_v2` or changes the
implicit production default.

The corrected analyzer and records passed 31 targeted tests.  The complete
local suite passed 1,942 tests and 8 subtests with no failure, skip,
deselection, or xfail.  `git diff --check` also passed.
