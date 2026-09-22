# Phase 6 P6.1: H100 qualification support

Classification: `READY_P6_1_H100_CANDIDATE_GATE`.

Baseline: Phase 5 closure commit
`9570c8c7c7d6bacf22ee09ab7dcaeff69f9709af`.

The user explicitly authorized Phase 6 execution.  P6.1 supplies the narrow,
auditable measurement and aggregation layer needed to execute the already
frozen H100 contract.  It does not alter either timestep implementation,
change the production default, authorize a long run before the candidate
gate passes, or change any scientific equation or parameter.

## Production-runtime profiler

`benchmarks/profile_plane_runtime_timestep.py` constructs either
`legacy_production` or `compiled_v2` through its real production builder.
Both paths are measured with the same outer whole-timestep timer and the same
initial tensor generator.  Each report binds the requested and effective
runtime identity, Git identity, device, dtype, TF32 state, frozen numerical
configuration, transform-call counts, Dynamo graph-break counters, CUDA peak
memory, initial-Q SHA-256, final-state SHA-256, finite-state result, trial
number, and completed-step count.

Profiling deliberately disables the periodic spectral refresh so the frozen
7-forward/32-inverse per-step oracle is measurable without refresh cadence
contamination.  Production trajectory and restart gates retain the ordinary
production refresh policy.

## Evidence aggregation

`scripts_plane/analyze_phase6_compiled_v2_qualification.py` is analysis-only.
It requires exactly twelve profiles: two grids, two runtime paths, and three
paired trials.  It also requires exactly six successful, byte-identical
workflow comparisons: one cross-runtime 100-step comparison and two
same-runtime restart comparisons per grid.  Paired profiler runs must share
their initial-Q and final-state identities.

The auxiliary gate report is mandatory.  Its exact gate set covers the two
CUDA-only tests, COMPLETE semantics, diagnostic byte identity, normalized
metadata identity, filename/shape/dtype identity, both directions of
cross-runtime restart rejection, tampered-identity rejection for both
runtimes, and pre/post worktree cleanliness.  The report must bind one
successful H100 Slurm submission, the candidate commit, and TF32-off state.

The aggregator fails closed on missing, duplicated, fallback, dirty,
non-finite, graph-break, transform-count, numerical, performance, memory, or
auxiliary evidence.  Passing it makes the separately frozen T=400 long-run
stage eligible; it never promotes a default.

## Local validation boundary

The profiler and aggregator have CPU synthetic coverage, including both real
runtime constructions, exact short-run state identity, strict evidence-matrix
cardinality, performance failure, fallback, commit, transform-call,
final-state, comparison, auxiliary-gate, and no-overwrite failures.  A local
RTX 3060 Ti functional smoke additionally confirmed distinct effective
runtime paths, equal initial and final hashes, finite output, and 7/32
transform calls.  This local smoke is not an H100 performance result.

The final P6.1 targeted set passed 31 tests.  The complete local suite passed
1,936 tests and 8 subtests with no failure, skip, deselection, or xfail.
`git diff --check` also passed.

The formal H100 candidate gate must run from a clean detached worktree at the
exact P6.1 commit.  The separately authorized long-run stage remains blocked
until that candidate gate is complete and classified PASS.
