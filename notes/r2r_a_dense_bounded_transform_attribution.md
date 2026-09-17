# R2R-A: dense bounded-transform attribution

## Purpose

R2R-A starts an isolated post-v0.1 performance investigation of the
cell-centered DCT-II and DST-II axes. The frozen PSSolver 0.1.0 release remains
unchanged. This stage adds measurement capability only:

- no FFT-based real-to-real candidate exists yet;
- no transform basis, normalization, mode ordering, PDE, or boundary condition
  changes;
- the dense orthonormal matrix implementation remains the only implementation
  and the production default;
- no result from this stage authorizes a default change.

The work is isolated on **perf/bounded-r2r-fft-v0.1**, based directly on the
**v0.1.0** tag.

## Current scaling contract

For a bounded axis of physical size N and retained modal size R, the current
matrix product performs work proportional to

\[
    N_{\rm lines} N R.
\]

A full transform has R=N, while the qualified projected-transform path may use
R<N. The transform matrix cache still stores the complete real N by N
orthonormal matrix. In the Plane geometry only the wall-normal axis is bounded;
the two in-plane axes remain periodic cuFFT axes.

This is a scaling limitation, not a spectral-accuracy defect.

## Complete-timestep attribution

The complete Beris--Edwards profiler still reports inclusive
“transform_forward” and “transform_inverse” timings. Passing the explicit
“--transform-attribution” flag adds nested aggregate regions:

- “periodic_fft_forward”
- “periodic_fft_inverse”
- “bounded_dct_forward”
- “bounded_dct_inverse”
- “bounded_dst_forward”
- “bounded_dst_inverse”

Each aggregate also has an execution-detail region:

- “_axis” for one-axis complex FFT calls;
- “_nd” for the Plane rfftn/irfftn calls;
- “_full” for a complete DCT/DST matrix product;
- “_truncated” for a retained-row DCT/DST matrix product.

These regions are nested inside the existing transform regions and must not be
added to the additive timestep partition. CUDA events remain asynchronous;
there is no synchronization inside a timestep.

Attribution is disabled by default. This keeps the ordinary profiler suitable
for throughput comparisons without the overhead of the additional CUDA event
pairs. R2R-A attribution runs and ordinary throughput runs should therefore be
recorded separately.

The sole production-code change is a narrow method around the already
existing torch.fft.rfftn and torch.fft.irfftn calls so benchmark tooling can
time them. The operations, arguments, order, and returned tensors are
unchanged.

## Standalone dense baseline sweep

Run a small local sweep with:

~~~bash
python -m benchmarks.benchmark_bounded_axis_transforms \
  --device cpu \
  --dtype float64 \
  --sizes 20,40,80,120,160 \
  --line-count 256 \
  --warmup 3 \
  --repeats 10 \
  --output /tmp/r2r_a_dense_cpu.json
~~~

For a CUDA sweep:

~~~bash
python -m benchmarks.benchmark_bounded_axis_transforms \
  --device cuda \
  --dtype float64 \
  --sizes 40,80,120,160,240,320 \
  --kinds dct,dst \
  --execution-modes full,truncated \
  --value-types real,complex \
  --retained-fraction 0.5 \
  --line-count 4096 \
  --warmup 5 \
  --repeats 20 \
  --output /tmp/r2r_a_dense_cuda.json
~~~

The sweep records forward and inverse timing distributions, consistency
errors, retained sizes, a dense scalar-product work model, actual cached
matrix bytes, the projected two-matrix Plane cache size, and CUDA allocator
peaks when applicable.

The benchmark never writes into a scientific simulation directory and refuses
to replace an existing output unless “--overwrite” is explicit.

## Local qualification

The initial R2R-A implementation passed:

- 80 focused benchmark, profiler, transform-order, and dtype tests;
- the complete CPU suite with 1035 tests and eight subtests;
- a 16-case float64 CUDA smoke covering N=40/80, DCT/DST,
  full/truncated, and real/complex values on an RTX 3060 Ti.

Frozen-v0.1 and R2R-A Neumann/Dirichlet forward and inverse hashes were
byte-identical for a deterministic Hermitian-half transform comparison.
All CUDA cases were finite. The largest coefficient-recovery relative L2 was
9.28e-15. A separate two-step Plane CUDA profiler smoke emitted all six
aggregate periodic/DCT/DST regions and the expected Hermitian/truncated detail
regions.

These are implementation and observability checks, not a production-scale
performance conclusion. The H100 scaling sweep remains future R2R evidence.

## Interpretation boundary

R2R-A can establish:

1. the isolated dense DCT/DST scaling curve;
2. full versus retained-row cost;
3. real versus complex matrix-product cost;
4. the fraction of a complete Plane timestep spent in periodic FFT, DCT, and
   DST regions;
5. the measurements a later FFT candidate must beat.

R2R-A cannot establish:

- that an FFT DCT/DST candidate is correct;
- the crossover size between dense and FFT implementations;
- eligibility for a hybrid policy;
- eligibility for any production promotion.

Those decisions belong to R2R-B through R2R-F.
