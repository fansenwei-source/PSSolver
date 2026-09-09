# Initial performance baseline

Date: 2026-09-08

Baseline commit: `d32008949a0024dbe5db9ce64ed5881665ecbb64`

Local environment:

- GPU: NVIDIA GeForce RTX 3060 Ti, 8 GiB
- Python: 3.12.13
- PyTorch: 2.6.0
- CUDA runtime reported by PyTorch: 12.6
- Formal benchmark arithmetic: float64 / complex128

## Regression baseline

The unmodified worktree completed:

```text
415 passed, 8 subtests passed in 13.07s
```

## Rejected experiment: adjacent periodic FFT fusion

Replacing adjacent axis-wise FFT calls with one `fftn` call was numerically
equivalent but did not provide a robust performance benefit:

| Device | Shape | Boundary conditions | Legacy/experimental speedup |
|---|---:|---|---:|
| CPU | 64x64x32 | periodic/periodic/Neumann | 0.662x |
| RTX 3060 Ti | 64x64x32 | periodic/periodic/Neumann | 1.048x |
| RTX 3060 Ti | 320x320x80 | periodic/periodic/Neumann | 1.010x |

The experiment was removed. The result is retained here to prevent the same
optimization from being accepted later without hardware-specific evidence.

## Accepted candidate: optional pressure residual diagnostics

The production Beris--Edwards driver previously evaluated two vector norms,
performed two GPU-to-host scalar synchronizations, and applied an additional
pressure residual operator every timestep. Those values are only consumed when
diagnostics are enabled.

The candidate keeps the pressure solution unchanged and preserves the previous
behavior when diagnostics are enabled. Production mode omits only the unused
residual measurements.

| Device | Shape | Diagnostics/production Stokes speedup | Solution max abs difference |
|---|---:|---:|---:|
| CPU | 64x64x32 | 1.787x | 0 |
| RTX 3060 Ti | 64x64x32 | 2.056x | 0 |
| RTX 3060 Ti | 320x320x80 | 1.884x | 0 |

A separate 100-step, 32x32x16 CUDA driver run measured:

- baseline elapsed time: 0.709154 s;
- candidate elapsed time: 0.682542 s;
- end-to-end speedup: 1.039x;
- final `Q_100.npy` SHA-256 for both runs:
  `5718ffa656039ec97d935b04692ceae91dfd10926f4c14b6fb6dc4820d4ab31f`.

The final candidate worktree completed:

```text
422 passed, 8 subtests passed in 12.80s
```

## Interpretation boundary

The approximately 1.9x number applies only to the modal Stokes subsolve. It
does not imply a 1.9x complete timestep speedup. The small-grid driver result
suggests a roughly 4% end-to-end improvement on this local GPU. A representative
H100 profile and repeated whole-timestep benchmark are required before merge.

## Complete-timestep profile

The benchmark-only profiler uses the production Beris--Edwards nonlinear
model, complete nematic force, mixed-basis Stokes solver, and `cubic_half`
projection. It uses asynchronous CUDA events and does not synchronize inside
the timestep. Spectral refreshes and snapshot output were disabled for the
pure-compute measurements.

| Shape | Mean timestep | Timesteps/s | Static fields | Q nonlinear | Nematic force | Inverse transforms | Peak allocated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32x32x16 | 7.664 ms | 130.48 | 63.29% | 32.45% | 54.63% | 54.12% | 28.1 MiB |
| 64x64x32 | 34.399 ms | 29.07 | 66.98% | 27.46% | 59.18% | 55.21% | 170.7 MiB |
| 128x128x32 | 136.604 ms | 7.32 | 67.12% | 27.37% | 59.48% | 54.76% | 646.7 MiB |

Static fields, Q nonlinear, IMEX/dealiasing, dynamic inverse, and spectral
refresh form the additive timestep partition. Nematic force and transform
measurements are nested cross-cutting regions, so their percentages must not
be added to the partition percentages.

At 128x128x32, the modal Stokes solve itself is only 2.04% of the timestep.
The complete nematic-force construction is the dominant model region, while
57 inverse transforms per step account for 54.76% of the timestep across the
static and nonlinear paths. The next optimization study should therefore
target repeated derivative/inverse-transform work in force construction and
Q dynamics, rather than further pressure-solver tuning.

## Accepted candidate: single-step Q-gradient reuse

The complete nematic force and Q nonlinear model previously evaluated the same
15 Q-component gradients independently within one timestep. The candidate
stages the gradients computed by the static model and permits one guarded read
by the immediately following nonlinear evaluation. It checks both spatial and
spectral tensor version counters; an intervening field mutation invalidates the
entry and preserves the original recomputation semantics. A driver switch,
`--disable-q-gradient-reuse`, provides an otherwise identical uncached control.

| Shape | Uncached timestep | Cached timestep | Speedup | Inverse calls/step | Final state |
|---:|---:|---:|---:|---:|---:|
| 32x32x16 | 7.284 ms | 6.587 ms | 1.106x | 57 to 42 | exact |
| 64x64x32 | 34.590 ms | 30.434 ms | 1.137x | 57 to 42 | exact |
| 128x128x32 | 136.450 ms | 120.548 ms | 1.132x | 57 to 42 | exact |

The cached and uncached peak allocated memory was identical at 128x128x32
(678,099,456 bytes). At 64x64x32 the cached path differed by only 128 KiB.

A separate production-driver control compared commit `f86a6ec` with this
candidate for 100 CUDA timesteps at 32x32x16. Elapsed time changed from
0.677329 s to 0.573871 s, a 1.180x single-run speedup. Final Q, velocity, and
pressure arrays were byte-identical, with SHA-256 values:

- Q: `5e1cca8c1a4a4f8f518cdda626aefd2aa1c2e0ba68e51543f450ace9d0c2e92b`;
- velocity: `eb2da6a2abf3a6497de06e2cf3a2aa0d7577d42007d450a327f353398094b213`;
- pressure: `7b43829476181594c573240baee794a695a32b64149ce3f4e8fbca548c8afb8d`.

These local figures establish a useful candidate, not an H100 production
claim. The current complete regression result is:

```text
436 passed, 8 subtests passed in 13.51s
```

## Rejected experiment: same-basis derivative batching

A follow-up candidate grouped Q Laplacians, Q gradients, and the two
tangential-velocity gradients that share one input basis and derivative axis.
It changed no transform definitions and reduced inverse-transform calls from
42 to 23 per timestep. Batched and separate paths produced identical final
state hashes, and peak allocated memory was identical at every tested shape.

Each entry below is the median of three alternating batched/separate runs with
five warmup and 30 measured timesteps:

| Shape | Separate timestep | Batched timestep | Separate/batched |
|---:|---:|---:|---:|
| 32x32x16 | 6.541 ms | 5.616 ms | 1.165x |
| 64x64x32 | 30.109 ms | 29.973 ms | 1.005x |
| 128x128x32 | 120.459 ms | 121.465 ms | 0.992x |

The small-grid kernel-launch benefit disappeared as transform size increased;
the largest case was approximately 0.8% slower. The candidate therefore failed
the requirement for stable benefit on the medium and large local grids. All
production, benchmark, CLI, and test changes from this experiment were removed.

## Nsight Systems CUDA-kernel profile

Nsight Systems 2025.1.3 captured five timesteps after five warmup steps, with
CUDA event completion tracing disabled. The workload used float64/complex128,
`cubic_half`, Q-gradient reuse, no spectral refresh, and no snapshots. The
capture runner exposes the complete-timestep profiler regions as nested NVTX
ranges and brackets only the measured steps with the CUDA profiler API.

| Shape | Mean timestep | GPU ops/step | Inverse transforms | Forward transforms | Nematic force | Q nonlinear | Stokes solve |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64x64x32 | 30.487 ms | 1076 | 49.8% | 23.0% | 67.6% | 17.4% | 2.3% |
| 128x128x32 | 120.974 ms | 1076 | 49.5% | 23.7% | 67.4% | 17.8% | 2.3% |

The transform percentages are GPU time projected into non-overlapping forward
and inverse NVTX ranges. Nematic force and Q nonlinear are higher-level nested
ranges and therefore must not be added to the transform percentages.

| Shape | CUTLASS complex GEMM | FFT kernels | Explicit GPU memops | Kernel-launch API time / GPU range |
|---:|---:|---:|---:|---:|
| 64x64x32 | 50.1% | 15.7% | 0.33% | 5.8% |
| 128x128x32 | 45.6% | 18.3% | 0.39% | 1.5% |

The CUTLASS kernels come from the dense matrix multiplication used for the
cell-centered DCT/DST axis in `TensorProductTransformBackend`. FFT kernels
implement the periodic axes. The result explains why reducing launch count by
same-basis batching helped only the smallest grid: launch overhead becomes
minor at 128x128x32, while transform arithmetic remains dominant.

The next local optimization study should therefore evaluate an FFT-based
orthonormal DCT-II/DCT-III and matching DST implementation, while preserving
the current cell-centered bases, modal indexing, normalization, derivative
parity, and terminal-mode behavior. This is a transform-algorithm change and
requires manufactured transform/derivative tests plus complete trajectory
comparison; it must not be inferred safe from timing alone. The RTX 3060 Ti
ranking is also not an H100 performance claim because their float64 GEMM
throughput differs substantially.

The Nsight and direct-control runs produced identical final-state SHA-256
values at each shape:

- 64x64x32:
  `c975374a3729249e410ce91ec982487bb9fb2234ee55255a9df6760ead39e4ef`;
- 128x128x32:
  `7d479d97cac48de41e3d8ce9e58215e8fda9d2035007d4f8f3db1523921e140b`.

## Rejected experiment: FFT-based cell-centered DCT/DST

Two default-off alternatives to the dense orthonormal cell-centered DCT-II,
DCT-III, DST-II, and DST-III transforms were implemented and evaluated. The
first used a length-`2N` complex FFT of an even or odd extension. The second
split complex inputs into real and imaginary parts and used batched
`rfft`/`irfft` half-spectra. Both preserved the existing modal indexing,
normalization, mixed-boundary derivative maps, and terminal DST behavior.

The complex-FFT implementation passed 501 tests and eight subtests, including
new real/complex transform comparisons in float32 and float64, sizes from one
through 32, mixed three-dimensional bases, analytic derivatives, and the full
manufactured free-slip Stokes solution. A 32x32x16 float64 CUDA trajectory
comparison through 100 timesteps found no growing discrepancy. At step 100,
the complete spatial-state relative L2 difference was `4.76e-16`; Q-component
relative differences were below `9.0e-16`, velocity-component differences were
below `5.4e-15`, and the pressure difference was `1.45e-14`.

Each timing entry below is the mean of three runs with five warmup and 30
measured timesteps. The matrix control was rebuilt for every comparison.

| Shape | Matrix timestep | Complex-FFT timestep | Matrix/FFT | Matrix/FFT peak allocated |
|---:|---:|---:|---:|---:|
| 64x64x32 | 30.330 ms | 29.663 ms | 1.023x | 170.8 / 235.8 MiB |
| 128x128x32 | 121.255 ms | 118.480 ms | 1.023x | 646.7 / 910.7 MiB |

Although the complex-FFT form was consistently about 2.3% faster on the local
RTX 3060 Ti, peak allocated memory increased by approximately 38% and 41% at
the two sizes. The half-spectrum implementation did not fix this tradeoff:

| Shape | Matrix timestep | Half-spectrum timestep | Matrix/FFT | Matrix/FFT peak allocated |
|---:|---:|---:|---:|---:|
| 64x64x32 | 30.269 ms | 34.672 ms | 0.873x | 170.8 / 271.8 MiB |
| 128x128x32 | 120.734 ms | 140.351 ms | 0.860x | 646.7 / 1059.2 MiB |

The half-spectrum version was 14.5--16.2% slower and used still more memory,
because separating complex data created large temporary real batches. The
best variant therefore offered too little local speedup for its memory cost,
especially for the planned large three-dimensional grids. All solver,
profiler, CLI, and test changes from both variants were removed. This does not
rule out a native vendor DCT/DST implementation or a custom fused kernel, and
it is not an H100 ranking; it rejects these two PyTorch composition strategies
for the present solver.

## Accepted candidate: real-first tensor-product execution

The transform algorithm and full-complex spectral layout can remain unchanged
while avoiding most complex DCT/DST matrix products. Transforms on independent
axes commute, so the candidate executes all DCT/DST axes before periodic FFT
axes in the forward transform. The inverse applies periodic inverse FFTs first,
takes the real part that the public inverse already returns, and then applies
the real-basis inverse matrices. The legacy order remains available explicitly, while `real_first` is
exposed through the solver, profiler, Nsight runner, and production driver and is
recorded in metadata.

The final candidate passed 467 tests and eight subtests. Coverage included
float32/float64 forward and inverse
equivalence, arbitrary complex inverse coefficients, periodic/Neumann/
Dirichlet axis permutations, mixed-basis derivatives, full manufactured
free-slip Stokes recovery, force, pressure, and energy-budget tests. A
32x32x16 float64 CUDA comparison through 100 complete timesteps found no error
growth. At step 100, the complete spatial-state relative L2 difference was
`3.32e-16`; Q-component differences were below `3.9e-16`, velocity-component
differences below `5.6e-15`, and pressure `8.3e-15`.

Each profiler entry below is the mean of three alternating legacy/real-first
runs with five warmup and 30 measured timesteps:

| Shape | Legacy timestep | Real-first timestep | Speedup | Legacy/real-first peak allocated |
|---:|---:|---:|---:|---:|
| 64x64x32 | 30.440 ms | 19.084 ms | 1.595x | 170.8 / 152.8 MiB |
| 128x128x32 | 121.256 ms | 79.693 ms | 1.522x | 646.7 / 582.7 MiB |

Forward-transform time fell from 6.928 to 3.358 ms at 64x64x32 and from
28.354 to 13.948 ms at 128x128x32. Inverse-transform time fell from 15.376 to
8.332 ms and from 60.457 to 36.067 ms, respectively. Peak allocated memory
fell by 10.5% and 9.9%. Peak reserved memory was 220/236 MiB at the smaller
shape and 948/868 MiB at the larger shape; allocator reservation is therefore
not used as the acceptance metric.

A matched production-driver run at 32x32x16 for 100 float64 CUDA timesteps
completed in 0.5732 s with the legacy order and 0.4973 s with real-first, a
1.153x end-to-end speedup. Final Q, velocity, and pressure relative L2
differences were `3.09e-16`, `4.25e-16`, and `4.72e-16`.

A bounded five-step 128x128x32 Nsight capture confirmed the intended kernel
change. Legacy complex CUTLASS GEMMs accounted for approximately 45.6% of GPU
time. With real-first, the dominant real double-precision CUTLASS GEMMs
accounted for approximately 17.7%, while the remaining complex GEMMs were
approximately 1.4%. Mean captured timestep time was 79.575 ms, inverse
transform time was about 36.0 ms, and forward transform time about 13.8 ms.

This is a strong candidate because it changes only execution order, not
the basis, normalization, modal indexing, spectral shape, derivative maps,
dealiasing masks, Stokes equations, or Beris--Edwards model. The historical
order remains available as an explicit `legacy` fallback.

### H100 qualification and default decision

A clean H100 PCIe comparison at commit `4bbcb219` used three alternating
legacy/real-first trials per shape, five warmup steps, and 30 measured complete
timesteps. The 320x320x80 production grid met every predeclared acceptance
criterion:

| Shape | Legacy timestep | Real-first timestep | Speedup | Allocated-memory ratio |
|---:|---:|---:|---:|---:|
| 128x128x32 | 14.457 ms | 13.439 ms | 1.076x | 0.905 |
| 320x320x80 | 204.772 ms | 176.129 ms | 1.163x | 0.899 |

All three 320x320x80 real-first trials were faster than their paired legacy
controls. Peak allocated memory fell by 10.12% and peak reserved memory fell
by 7.21%; no OOM, NaN, Inf, or configuration mismatch occurred. The full CPU
suite completed with 466 passed tests, one optional-dependency skip, and eight
passed subtests. Together with the transform, manufactured-solution, and
100-step CUDA trajectory comparisons above, this qualifies `real_first` as
the production default. `legacy` remains an explicit compatibility and
diagnostic option.

The post-promotion local gate completed 68 targeted tests and the full suite
with 472 passed tests and eight passed subtests. Three matched 32x32x16,
100-step, float64 CUDA driver runs compared the new implicit default, explicit
`real_first`, and explicit `legacy`. The implicit default and explicit
`real_first` Q, velocity, and pressure files were byte-identical. Relative L2
differences from `legacy` were `3.04e-16`, `4.35e-16`, and `4.38e-16`,
respectively, with finite values throughout.

### Candidate: spectral-linear molecular-field staging

The complete-stress path previously inverse-transformed each of the five
`L1 laplacian(Q)` components, assembled the raw molecular field in physical
space, and immediately transformed that field back to spectral space for
projection. Linearity permits the local A/B/C bulk contribution to remain in
physical space while adding the `L1 laplacian(Q)` contribution directly to its
spectral transform. The candidate removes five inverse transforms per
timestep without changing the equation, basis, normalization, modal indexing,
projector, or dealiasing rule. The existing physical path remains the default
pending H100 qualification.

Three alternating 128x128x32 float64 CUDA pairs on the RTX 3060 Ti measured:

| Path | Mean timestep | Nematic force | Inverse calls/step | Peak allocated |
|---|---:|---:|---:|---:|
| Physical linear term | 79.499 ms | 51.615 ms | 42 | 585,824,768 B |
| Spectral linear term | 76.688 ms | 48.749 ms | 37 | 585,824,768 B |

The mean and median speedups were `1.0367x` and `1.0373x`; all three paired
runs favored the candidate. Shortening the lifetime of physical Laplacian
temporaries also reduced peak allocated memory from the pre-candidate baseline
of 610,990,592 bytes to 585,824,768 bytes for both paths.

A matched 32x32x16 production-driver comparison through 100 float64 CUDA
timesteps found relative L2 differences of `3.92e-17` for Q, `2.15e-16` for
velocity, and `3.45e-16` for pressure. The candidate branch's default physical
path remained byte-identical to commit `57cf8b7`. The final candidate suite completed with 477 passed tests and eight passed
subtests.

## Snapshot I/O profile

A separate 64x64x32 run saved Q, velocity, and pressure after every profiled
step. The mean transfer plus write cost was 14.79 ms per snapshot, compared
with 38.74 ms of numerical work in that short run. This is significant for
every-step output, but approximately 0.043% of compute time when the same
snapshot cost is amortized over a 1000-step save interval.

Raw JSON results were written outside the repository:

- `/tmp/pssolver_profile_be_32_20260908.json`
- `/tmp/pssolver_profile_be_64_20260908.json`
- `/tmp/pssolver_profile_be_128_20260908.json`
- `/tmp/pssolver_profile_be_io_64_20260908.json`
- `/tmp/pssolver_qgrad_cached_32_20260908.json`
- `/tmp/pssolver_qgrad_uncached_32_20260908.json`
- `/tmp/pssolver_qgrad_cached_64_20260908.json`
- `/tmp/pssolver_qgrad_uncached_64_20260908.json`
- `/tmp/pssolver_qgrad_cached_128_20260908.json`
- `/tmp/pssolver_qgrad_uncached_128_20260908.json`
- `/tmp/pssolver_batched_derivatives_batched_<shape>_trial{1,2,3}_20260908.json`
- `/tmp/pssolver_batched_derivatives_separate_<shape>_trial{1,2,3}_20260908.json`

For the derivative-batching files, `<shape>` is one of `32x32x16`,
`64x64x32`, or `128x128x32`.

Final Nsight artifacts and CSV summaries were generated from clean commit
`05275fad0d927d553fa3d95d0e2b6e82927559f9`:

- `/tmp/pssolver_nsys_be_64x64x32_05275fa_clean_20260908.nsys-rep`;
- `/tmp/pssolver_nsys_be_64x64x32_05275fa_clean_20260908.json`;
- `/tmp/pssolver_nsys_be_64x64x32_05275fa_clean_20260908_stats_*.csv`;
- `/tmp/pssolver_nsys_be_64x64x32_05275fa_clean_direct_20260908.json`;
- `/tmp/pssolver_nsys_be_128x128x32_05275fa_clean_20260908.nsys-rep`;
- `/tmp/pssolver_nsys_be_128x128x32_05275fa_clean_20260908.json`;
- `/tmp/pssolver_nsys_be_128x128x32_05275fa_clean_20260908_stats_*.csv`;
- `/tmp/pssolver_nsys_be_128x128x32_05275fa_clean_direct_20260908.json`.

The rejected FFT DCT/DST experiment wrote its repeated local-GPU profiles to:

- `/tmp/pssolver_fft_ab.mb4XU3/` for the length-`2N` complex-FFT variant;
- `/tmp/pssolver_fft_half_ab.uPwKAc/` for the batched half-spectrum variant.

The accepted real-first experiment wrote local artifacts to:

- `/tmp/pssolver_real_first_ab.ztJanq/` for repeated profiler A/B results;
- `/tmp/pssolver_real_first_driver.HCgtZd/` for matched production-driver runs;
- `/tmp/pssolver_nsys_be_128x128x32_real_first_20260909.nsys-rep`;
- `/tmp/pssolver_nsys_be_128x128x32_real_first_20260909.json`;
- `/tmp/pssolver_nsys_be_128x128x32_real_first_20260909_stats_*.csv`.
