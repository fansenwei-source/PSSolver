# Phase 9 P9.5: H100 functional forward, VJP, and memory qualification

## Decision

P9.5 qualifies the existing periodic complete-stress functional runtime on one
H100 for **batch size one only**.  It does not add a batch dimension merely to
create a broader claim.  Larger batches remain an optional future extension
after a real consumer demonstrates that they are needed.

The machine-readable frozen contract is
[phase_9_p95_h100_qualification_plan.json](phase_9_p95_h100_qualification_plan.json).
It must be committed before any formal H100 evidence is collected.

## Matrix and identity

The frozen grids are R128 `(128, 128, 32)` and R320 `(320, 320, 80)`, both at
physical lengths `(100, 100, 20)`.  For each grid, three fresh processes run
each of:

1. production forward timesteps;
2. functional forward timesteps;
3. a functional reverse-mode VJP.

Three balanced-independent trials per role produce 18 profile JSON files.
All roles use the same immutable analytic `Q_0.npy`, spatially varying activity,
float64, eager pointwise execution, disabled spectral refresh, and TF32 off.
The input-preparation tool records the raw-file SHA-256; the analyzer requires
all roles in a grid to use that same identity.

## Correctness gates outside timing

The first functional-forward and VJP trial on each grid repeats the frozen R12
production/functional consistency check.  The first VJP trial also repeats the
P9.4 gradient and hidden-detach validator on the H100.  These checks run outside
the timed window.  Paired production and functional final-state hashes must be
exact, and functional state and VJP hashes must replay exactly across trials.

The CUDA-only tests excluded from a GPU-less login-node suite are listed
explicitly in the JSON plan.  Every one must run and pass on the allocated H100
before profiling starts; a login-node skip is not evidence.

## Performance and memory gates

The contract uses broad non-regression and feasibility gates, not a claim that
reverse mode is as cheap as forward execution.  Functional forward mean and
median time may be at most 1.20 times production; its peak allocated and
reserved memory may be at most 1.75 times production.  VJP mean time may be at
most 12 times functional forward.  VJP allocated and reserved peaks must remain
below 85% and 95% of the H100 device memory.  Each profile must have coefficient
of variation at most 0.20.

These limits distinguish an unusable functional path from ordinary profiler
noise while preserving exact numerical gates.  No limit may be edited after
the formal job begins.

## Evidence and stopping rules

The formal run permits one H100 Slurm submission and no automatic retry.  Every
profile is a fresh Python process and is atomically committed.  The analyzer is
fail-closed on missing, duplicate, dirty-worktree, wrong-commit, wrong-device,
non-finite, identity, correctness, performance, or memory evidence.  A
`COMPLETE` marker may be written only after the repository analyzer returns
`PASS_P9_5_H100_BATCH_ONE_FORWARD_VJP_MEMORY` and the archive checksum manifest
verifies.

P9.5 does not change the production runtime, defaults, stable package root,
optimizer ownership, objective ownership, or checkpoint scheduling.  It does
not qualify Channel pressure differentiation or an independent control
consumer.  P9.6 remains unauthorized until this qualification passes and its
result is reviewed and recorded.
