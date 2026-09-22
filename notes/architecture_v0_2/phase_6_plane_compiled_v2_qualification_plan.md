# Phase 6 frozen plan: compiled-v2 Plane qualification

Status: `FROZEN_NOT_AUTHORIZED`.

Phase 6 compares `legacy_production` and `compiled_v2` from the same exact
commit.  It must not mix architecture qualification with a change in
equations, boundary conditions, initialization, precision, transforms,
dealiasing, zero-mode policy, or time integration.

## Formal H100 qualification

One H100 job owns the complete balanced A/B gate.  It first runs all required
CUDA-only tests, then profiles R128 (`128x128x32`) and R320 (`320x320x80`).
Each grid uses three paired trials ordered A/B, B/A, A/B, with 10 warmup and
50 measured whole-timestep steps.  The candidate must preserve 7 forward and
32 inverse transforms per step, show no graph break or fallback, remain within
2 percent of baseline mean and median timestep and peak allocated/reserved
memory, and be faster in at least two of three paired trials per grid.

The same job runs matched 100-step production workflows at both grids.
Initial Q identity, Q/u/p and diagnostics byte identity, output names, shapes,
dtypes, COMPLETE semantics, and normalized metadata must pass.  Same-runtime
50+50 restart must match continuous execution byte for byte.  Cross-runtime
and tampered-identity restarts must fail before mutation.

Any incomplete output, dirty worktree, test failure, fallback, NaN, Inf, OOM,
CUDA error, transform-count mismatch, numerical mismatch, or threshold failure
stops the job.  There is no automatic retry.  A new attempt after an
environment-only failure requires explicit authorization and a new output
root.

## Long-run stage

Long-run execution is a separately authorized Phase 6 stage after the formal
H100 candidate gate.  Its frozen target is the A=18, R320, seed=24, dt=0.005,
T=400 workflow (80,000 steps), with saves every 1,000 steps and diagnostics
every 100 steps.  It compares stationarity, defect lines, sigma/H, sub-grid
sigma/H, and the wall-normal DCT spectrum against the matched legacy path and
requires the same initial-Q SHA-256.

This long-run stage determines whether the orchestration migration preserves
the accepted PSSolver benchmark trajectory.  It does not erase the already
documented difference between the PSSolver Stokes-inspired model and the full
inertial Shendruk model.

Neither formal H100 qualification nor the long-run stage is authorized by
this plan.  Even if both pass, changing the implicit production default needs
a separate decision and commit.
