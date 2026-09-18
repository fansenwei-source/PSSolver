# Bounded-axis applicability map: preliminary local stage-two report

> **Evidence status:** this is a local exploratory result, not an authoritative
> release qualification.  The raw maps were generated from the uncommitted
> stage-two worktree and were generated under
> `/tmp/pssolver_bounded_applicability_stage2_cIsZ5R`.  A future formal GPU
> qualification must run the documented command from a clean commit and bind
> its source revision and artifacts in durable storage.

## Scope

This report qualifies the benchmark infrastructure added after bounded-axis
execution commit `ea42c861eee77fe6dca7d142fc824330396f983f`.  The comparison
is between the production dense DCT/DST plan and a benchmark-local full-FFT
reference.  The reference is not pruned, is never imported by the solver and
cannot select or change a production algorithm.

The local measurements used an NVIDIA GeForce RTX 3060 Ti, PyTorch 2.6.0,
CUDA runtime 12.6, explicit TF32 off and the float64 or float32 dtype stated
below.  Every result passed finite-value, dense-reference, coefficient-space
identity and (for full transforms) roundtrip gates.  JSON was published last
as the completion artifact and binds the corresponding CSV hash and row count.

## Core maps

The core maps covered:

```text
N                 = 32, 80, 160, 320
R/N               = 1, 1/2, 1/8
line count        = 256, 4096
kind              = DCT, DST
direction         = forward, inverse
value type        = real, complex
trials/repeats    = 2 / 5
```

For float64, the full-FFT reference was faster in 90 of 192 cases and at
least 15% faster in 83.  It was at least 15% faster in both paired trials in
82 cases.  Only 9 of the 64 `R/N <= 1/8` cases reached the 15% threshold, and
those wins were confined to complex values.  The maximum dense-reference
relative L2 difference was `2.72e-14`.

For float32, full FFT was faster in only 1 of 192 cases and that case passed
the 15% threshold in both trials.  No `R/N <= 1/8` case passed.  The maximum
dense-reference relative L2 difference was `2.06e-5`, within the audited
size-aware float32 tolerance.

## Plane production anchors

The Plane anchors use the actual real-first bounded-axis sizes and approximate
group line counts.  A speedup below one means full FFT is slower than dense.

| Anchor | Direction | Dense ms | Full FFT ms | FFT/dense speedup | Dense peak delta | FFT peak delta |
|---|---:|---:|---:|---:|---:|---:|
| R128 Q DCT, N=32, R=16, L=81920 | forward | 0.717 | 1.188 | 0.603 | 10.0 MiB | 100.8 MiB |
| R128 Q DCT, N=32, R=16, L=81920 | inverse | 0.412 | 1.875 | 0.219 | 20.0 MiB | 114.6 MiB |
| R128 normal DST, N=32, R=15, L=16384 | forward | 0.171 | 0.305 | 0.561 | 1.9 MiB | 24.6 MiB |
| R128 normal DST, N=32, R=15, L=16384 | inverse | 0.087 | 0.460 | 0.189 | 4.0 MiB | 26.6 MiB |
| R320 Q DCT, N=80, R=40, L=512000 | forward | 21.386 | 23.500 | 0.910 | 156.2 MiB | 1562.5 MiB |
| R320 Q DCT, N=80, R=40, L=512000 | inverse | 26.050 | 31.187 | 0.834 | 312.5 MiB | 1738.3 MiB |
| R320 normal DST, N=80, R=39, L=102400 | forward | 4.394 | 5.342 | 0.823 | 30.5 MiB | 375.0 MiB |
| R320 normal DST, N=80, R=39, L=102400 | inverse | 5.311 | 7.062 | 0.752 | 62.5 MiB | 410.2 MiB |

All paired trials favored dense for every Plane anchor.  Full FFT also needed
roughly 5.6--13 times the temporary allocated memory, despite its smaller
persistent phase cache.

## Long-axis probe

A separate float64, real-valued probe covered `N=512,1024`, `R/N=1/8,1/16`,
line counts 256 and 4096, both transforms and both directions.  Full FFT was
faster in 12 of 32 cases and at least 15% faster in 9; all nine passed in both
paired trials.  The useful region was narrow:

- `N=512` did not reach the 15% threshold;
- `N=1024, R/N=1/8` usually favored full FFT, although large-line DST inverse
  reached only `1.13x`;
- at `N=1024, R/N=1/16`, forward could win for the smaller line count, while
  inverse consistently lost and the larger-line forward cases also lost.

This non-monotonic behavior is expected: the dense retained transform becomes
cheaper as R decreases, while the full FFT continues to compute all N modes.

## Decision

The present Plane workloads (`N=32/80`, real-first, `R/N` near one half) do
not justify replacing dense DCT/DST, and the old full-FFT route remains
rejected.  A true pruned implementation is not authorized by these results.

For a future general geometry, a pruned prototype is worth reconsidering only
when an important workload has a long bounded axis near `N >= 1024`, a measured
retained fraction around `R/N <= 1/8`, and enough forward and inverse calls to
affect complete-timestep time.  That prototype must compute only the requested
modes, preserve the dense normalization and adjoint, and pass separate CPU,
GPU, memory and complete-solver gates.  Until such a workload exists, dense
remains both the production path and the reference oracle.
