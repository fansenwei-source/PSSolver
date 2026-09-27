# Phase 8 P8.5.6: finite-Q anchoring H100 qualification plan

Status: `READY_P8_5_6_H100_QUALIFICATION_NOT_EXECUTED`.

Implementation commit: `c8028845293247290a2e9aa51ce8847206768bfa`.
Frozen qualification source commit:
`74a5226f6798cc311d3b839624b85349e3647fe8`; it contains the plan record and
the explicit per-process TF32-off qualification policy on top of the core
implementation.  Its recorded baseline is the P8.5.5 record commit
`2c99942`.  P8.5.6 is a
qualification boundary for the existing finite-Q anchoring relaxation pilot;
it is not permission to connect Robin Q to the complete Beris--Edwards
timestep, public `Simulation` compiler, or a production runtime selector.

## Candidate implementation

The Robin eigenbasis is still constructed deterministically in CPU float64:
root finding, dense basis materialization, inverse construction, condition
number, and affine-lift solve happen once before evolution.  The immutable
operator tensors are then copied once to the requested CPU or CUDA device.
The timestep performs only device-local dense transforms, diagonal modal
division, physical reconstruction, validation, and clock updates.  It must
not run root finding, basis construction, matrix inversion, or factorization.

The cache identity now includes the concrete allocated device.  Five Q
components must share a float64 device.  Checkpoint files remain CPU NumPy
arrays, but loading explicitly places validated tensors on the target runtime
device before the aggregate pre-mutation restore gate.  A GPU checkpoint may
therefore restart only into the same GPU identity unless an explicit future
migration contract is introduced.

The qualification profiler compares the existing per-component scalar
reference loop with the aggregate five-component runtime.  These are two
execution organizations of the same P8.5.5 reference equation, not two
different physical models.  Both must start from identical Q and finish with
identical Q in every paired trial.

## Frozen CPU and CUDA gates

Before any formal H100 submission, run the complete CPU suite from a clean
detached worktree.  A login node without CUDA may deselect exactly these five
CUDA-only node IDs:

1. `tests/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity`
2. `tests/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory`
3. `tests/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite`
4. `tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device`
5. `tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact`

Only the existing optional `nematics3d` dependency skip is admissible.  All
five deselected tests must run on the allocated H100 before profiling and must
pass without skip, deselect, xfail, or fallback.

## Frozen H100 profile matrix

Use one H100 PCIe, float64, TF32 disabled, lengths `(100, 100, 20)`,
`dt=0.005`, seed `20260927`, five warmup steps, and thirty measured steps.
Run exactly twelve profiles:

- shapes `(128,128,32)` and `(320,320,80)`;
- trials 1, 2, and 3;
- modes `scalar_reference` and `aggregate_runtime`;
- paired order `reference/aggregate`, `aggregate/reference`, then
  `reference/aggregate`.

Every report must identify the exact clean qualification source commit and import the
installed qualification wheel rather than the source checkout.  Every report
must be finite, use `NVIDIA H100 PCIe`, keep TF32 off, report no fallback or
graph break, and prove that no construction, root solve, or matrix
factorization entered the timed loop.  Per measured step the exact operation
contract is ten forward transforms, five inverse transforms, five Helmholtz
solves, and zero Helmholtz applies.  The maximum sampled-basis condition
number must not exceed `1e10`.

For each grid and trial, paired initial and final Q SHA-256 identities must be
equal.  Aggregating the three paired trials for each grid:

- aggregate/reference mean timestep ratio must be at most `1.15`;
- aggregate/reference median timestep ratio must be at most `1.15`;
- maximum peak allocated-memory ratio must be at most `1.10`;
- maximum peak active-memory ratio must be at most `1.10`;
- maximum peak reserved-memory ratio must be at most `1.10`.

The repository analyzer
`benchmarks/analyze_finite_q_anchoring_qualification.py` is authoritative for
these twelve reports.  A missing, duplicate, malformed, shadow-imported,
dirty, numerically different, or over-limit report fails closed.

## Local evidence and limits

The implementation passed `71 passed, 1 deselected` across P8.5.1--P8.5.6;
the deselected test was only this not-yet-created plan record.  The complete
pre-record suite passed `2397 passed, 1 deselected, 8 subtests passed` with no
failure, skip, or xfail.  With the record present, the complete suite passed
`2398 passed, 8 subtests passed`.  A local CUDA R128 smoke produced finite output,
exact frozen operation counts, and no construction work in the timed loop.
Those results are development evidence, not the formal H100 result.

Exactly one formal H100 job is authorized.  No automatic retry, second job,
long run, public-runtime connection, production-default change, P8.6, or
Phase 9 work is authorized.  Write `COMPLETE` only after the CPU gate, five
real-H100 CUDA tests, twelve-profile analyzer, provenance, and checksum
manifest all pass.  Otherwise archive the precise failure and stop.
