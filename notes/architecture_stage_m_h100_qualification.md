# Architecture Stage M: bounded H100 shadow qualification

## Outcome and boundary

Stage M adds the first explicit GPU gate for the separated Plane architecture.
It does not replace the production driver, change a solver default, authorize a
long benchmark, or migrate Channel.  The existing
`Plane_beris_edwards_stokes.py` remains the reference implementation and the
new architecture remains an opt-in shadow.

The local part of this stage implements and CPU-tests the qualification
machinery.  Scientific and performance evidence requires one separately
authorized H100 job on HPCC; consequently this commit contains no H100 result
and makes no production-readiness claim.

## H100 reference contract

The Stage L public API remains CPU-only.  Stage M uses distinct entry points
whose production reference must record all of the following:

- completed `Plane_beris_edwards_stokes.py` execution;
- the complete Beris--Edwards nematic-stress Stokes model;
- CUDA execution on a device whose name contains the explicitly requested GPU
  name;
- float64 arithmetic and ineffective TF32;
- `save_start_step=0`, saved Q/u/p, and Q-gradient reuse.

Both the GPU recorded by the production reference and the GPU executing the
shadow are checked.  There is no fallback to CPU or to a different accelerator.

## Numerical gate

The read-only plan fixes a deliberately small `128 x 128 x 32`, six-step
trajectory at `dt=0.005`.  It saves steps zero through six and uses a
two-step spectral-refresh interval so the comparison crosses three refresh
boundaries.  Production and shadow share the exact production `Q_0.npy` and
the following explicit numerical policies:

- float64, TF32 off;
- `cubic_half` dealiasing;
- truncated projected transforms;
- Hermitian half-spectrum Plane storage with real-first execution;
- spectral molecular-field and stress-divergence assembly;
- compiled pointwise kernels;
- zero-mean tangential mode and Q-gradient reuse.

The existing read-only trajectory comparator checks configuration identity,
input identity, complete and matching Q/u/p frame sets, shapes, dtypes,
finite values, and input hashes before and after reads.  Q and velocity use raw
relative L2 errors; pressure uses the demeaned relative L2 error for its gauge.
The fixed scientific tolerance is `1e-10`.

## Performance and memory observations

Performance is measured separately from trajectory correctness.  The
production profiler gained one opt-in `--initial-q-path` argument so both
implementations start from the same immutable production Q snapshot.  This
changes neither its default synthetic input nor any production solver path.

The plan runs three balanced production/shadow pairs.  Each profile performs
three warm-up steps followed by ten CUDA-event-timed steps.  Snapshot I/O is
outside the timed region.  Mean and median timestep, paired speedups, and peak
allocated and reserved CUDA memory are reported independently.

Stage M deliberately has no speed or memory promotion threshold.  A successful
report means numerical equivalence and complete, comparable H100 observations;
it sets `eligible_for_production_promotion=false`.  A later architecture
decision may use the measured cost to choose between optimization, further
shadow testing, or continued isolation.

## Planning and execution safety

`scripts_plane/plan_plane_shadow_h100_qualification.py` is planning-only.  It
requires nonexistent independent control and scratch roots and emits exact argv
vectors without creating directories or touching CUDA.  It fixes the bounded
trajectory, balanced profiles, comparison, and final analysis commands.

The H100 execution must additionally verify the exact Git commit and a clean
detached worktree, use one new output namespace, submit only one job with
requeue disabled, and stop after any nonzero command, GPU mismatch, missing
completion artifact, configuration mismatch, non-finite value, or failed
relative-L2 gate.  Input JSON hashes are embedded in the final report; the
trajectory report already carries per-array source hashes.

## New opt-in surfaces

- `Plane_beris_edwards_shadow_h100.py`: bounded GPU trajectory entry point;
- `benchmarks/profile_plane_shadow_timestep.py`: migrated-runtime profiler;
- `scripts_plane/analyze_plane_shadow_h100_qualification.py`: numerical,
  timing, and memory evidence aggregation;
- `scripts_plane/plan_plane_shadow_h100_qualification.py`: read-only command
  planner.

Nothing in `pssolver.experimental` is imported by the production Plane or
Channel entry points.

## First H100 construction finding

The first scientific H100 attempt completed the six-step production reference
but stopped while constructing the shadow runtime.  An unindexed `"cuda"`
request was retained as the execution-context identity, whereas PyTorch
reported the newly allocated coordinate tensors on the concrete `cuda:0`
device.  Strict equality correctly rejected those two different device
objects before any shadow timestep ran.

The recovery keeps that strict check and fixes the experimental legacy-
assembly boundary instead: an unindexed CUDA request is resolved once through
`torch.cuda.current_device()` before the solver, transforms, projector, and
model context are built.  Explicit devices such as `cuda:2` and non-CUDA
devices pass through unchanged.  This changes no production solver path and
does not treat device inconsistency as acceptable.
