# Phase 6 P6.2: matched T=400 long-run architecture equivalence

Status: `READY_P6_2_LONG_RUN_HANDOFF`.

P6.1 has passed as `performance_equivalent`; P6.2 therefore owns the final
long-time architecture-equivalence gate.  It does not own a new model,
parameter scan, default promotion, or strict Shendruk reproduction claim.

## Frozen pair

Run `legacy_production` and `compiled_v2` from the same clean execution
commit.  Both use A=18, shape `320x320x80`, lengths `100x100x20`, seed 24,
`dt=0.005`, 80,000 steps, Q/u/p saves at step 0 and every 1,000 steps through
80,000, and diagnostics every 100 steps.  Float64, TF32-off, zero-mean with
zero friction, ordinary production spectral refresh, `cubic_half`, truncated
projected transforms, `real_first`, Hermitian-half storage, spectral molecular
field and stress divergence, pointwise compile, Q-gradient reuse, and saved
hydrodynamics remain frozen.

The pair must have equal raw and projected initial-Q SHA-256 identities.
Neither run may fall back, restart from another run, or write a checkpoint.
One mandatory execution-provenance record binds the clean execution commit,
the single successful H100 simulation Job, TF32-off environment, both absolute
run paths, and both validation-config SHA-256 identities.

## Strong equivalence gate

P6.2 requires byte identity for all 81 paired Q frames, all 81 paired velocity
frames, all 81 paired pressure frames, `diagnostics.npy`, and
`diagnostics.csv`.  Metadata must be identical after removing only the
explicit runtime-selection identity, its canonical configuration hash, the
validation-config hash, workflow runtime label, and measured elapsed time.

This is stronger than separately comparing stationarity, defect lines,
integer or sub-grid sigma/H, and the wall-normal DCT spectrum.  These
observables are deterministic functions of the saved Q sequence; complete Q
byte identity proves that the architecture migration supplies identical
inputs to each analysis.  The report records this implication explicitly.

Physical stationarity remains a separate benchmark interpretation.  Two
identical trajectories can both retain initialization memory or both remain
non-stationary.  P6.2 therefore closes architecture equivalence only.

## Failure and authorization boundaries

Missing frames, mismatched execution provenance, mismatched identities, metadata differences outside the
allowlist, non-finite arrays, incomplete markers, runtime fallback, output
overwrite, dirty worktrees, CUDA/OOM errors, or any byte mismatch fail closed.
There is no automatic retry.

A P6.2 pass marks Phase 6 complete and makes Phase 7 planning eligible.  It
does not authorize Phase 7 execution, promote `compiled_v2`, or change the
production default.  A separate user decision remains mandatory.

The local P6.2 implementation passed 34 targeted tests.  The complete suite
passed 1,958 tests and 8 subtests with no failure, skip, deselection, or
xfail.  JSON, Python compilation, archive-source, and `git diff --check`
gates also passed.
