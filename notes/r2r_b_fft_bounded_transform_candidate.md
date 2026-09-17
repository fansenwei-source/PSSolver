# R2R-B: opt-in FFT bounded-transform candidate

## Decision

R2R-B adds a numerically qualified FFT implementation of the existing
cell-centered orthonormal DCT-II and DST-II transforms. It is available only
through the explicit `bounded_transform_algorithm="fft"` selector. The dense
matrix implementation remains the default.

The candidate is retained because it removes the quadratic asymptotic scaling
and becomes faster for sufficiently long bounded axes. It is not promoted
because the current Plane production regime uses one short, truncated bounded
axis: for the representative local R128 case with `Nz=32`, the FFT path remains
slower and uses more peak allocated memory.

This is a performance conclusion, not a correctness limitation.

## Frozen mathematical convention

The dense implementation remains the reference. R2R-B does not change:

- cell-centered collocation points;
- DCT-II or DST-II mode ordering;
- orthonormal normalization;
- Neumann or Dirichlet boundary semantics;
- retained-mode selection;
- dealiasing, projection, or PDE equations.

For real DCT input, the candidate reorders even samples followed by reversed
odd samples, applies an N-point real FFT, and applies the DCT-II phase and
orthonormal scale. The inverse reconstructs the required Hermitian half
spectrum, applies an N-point inverse real FFT, and reverses the permutation.
The DST-II uses the exact alternating-sign and reversed-mode identity with the
same DCT kernel. Complex-valued inputs use a general mirrored 2N-point complex
FFT formula.

The FFT cache is O(N): scale, phase, modulation, and permutation vectors. The
dense reference cache is O(N squared) per transform kind. Both implementations
remain spectral transforms; only the algorithm used to evaluate the same
orthonormal basis has changed.

## Truncated transform semantics

The current candidate computes the complete FFT-based modal axis and then
slices the retained low modes. Its inverse pads omitted high modes with zero
before applying the inverse transform. This exactly matches the existing dense
retained-row operator, but it means the candidate currently has O(N log N)
work even when only R less than N modes are retained.

This distinction explains why the crossover for the production-style
half-retained path occurs later than the crossover for a full transform. A
future stage may investigate a genuinely pruned real-to-real transform, but it
must remain a separate candidate with its own correctness and performance
qualification.

## Interfaces and rollback

The selector is threaded through `TensorProductTransformBackend`,
`SpectralSolver`, and the complete Beris--Edwards timestep profiler. Accepted
values are `dense` and `fft`; the default is `dense`. The profiler records the
effective selection in its transform metadata.

The standalone bounded-axis benchmark accepts `--algorithms dense,fft` and
rejects a candidate timing result unless the transform first passes the dense
reference error gate. The dense path does not allocate an FFT real-to-real
cache, while the FFT path does not allocate dense transform matrices.

## Local qualification

Environment:

- NVIDIA GeForce RTX 3060 Ti;
- PyTorch 2.6.0 with CUDA 12.6 runtime;
- float64;
- 4096 independent transform lines;
- 5 warm-up calls and 12 timed repeats per direction.

The final standalone sweep covered 80 combinations across N=20, 40, 80, 160,
and 320; DCT/DST; full/half-retained; real/complex; and dense/FFT algorithms.
All outputs were finite. The maximum FFT-versus-dense relative L2 error was
2.706e-14, below the 5e-13 gate.

For real input, combined forward-plus-inverse speedup, defined as dense time
divided by FFT time, was:

| N | DCT full | DCT half-retained | DST full | DST half-retained |
|---:|---------:|------------------:|---------:|------------------:|
| 20 | 0.815 | 0.606 | 0.736 | 0.534 |
| 40 | 1.280 | 0.695 | 1.133 | 0.633 |
| 80 | 1.432 | 0.891 | 1.220 | 0.746 |
| 160 | 1.776 | 1.004 | 1.557 | 0.894 |
| 320 | 3.635 | 1.760 | 3.187 | 1.569 |

Thus the full real transforms cross over near N=40 on this device, whereas the
half-retained DCT is only neutral near N=160 and the half-retained DST crosses
between N=160 and N=320. These values are hardware- and batch-dependent; they
are evidence for a policy study, not universal thresholds.

At N=320, one float64 dense matrix occupies 819,200 bytes and separate DCT and
DST matrices occupy 1,638,400 bytes. The shared FFT cache recorded 20,480
bytes. Temporary FFT workspaces, rather than the persistent cache, determine
the candidate's runtime peak memory.

## Complete Plane check

A production-like local comparison used:

- shape 128 by 128 by 32 and lengths 100 by 100 by 20;
- float64, `cubic_half`, `truncated`, `real_first`, and `hermitian_half`;
- eager pointwise algebra to isolate the transform choice;
- 2 warm-up steps and 8 measured steps from the same deterministic initial
  condition.

Final FFT-versus-dense relative L2 errors were:

- Q: 1.092e-15;
- velocity: 6.252e-15;
- pressure: 2.327e-14.

The dense mean timestep was 41.604 ms and the FFT mean timestep was 58.836 ms,
so dense/FFT speedup was 0.707: the candidate was about 1.41 times slower.
Peak allocated memory was 400,599,552 bytes for dense and 434,205,696 bytes for
FFT, an FFT/dense ratio of about 1.084.

The implementation passed 135 focused tests and the complete local CPU suite
with 1,114 tests and eight subtests. Tests cover small, odd, and even sizes;
real and complex input; DCT and DST; full and truncated execution; mixed
tensor-product boundary conditions; Hermitian storage; autograd; selector
validation; metadata; and cache isolation.

## Promotion boundary and next evidence

R2R-B supports these conclusions:

1. the O(N log N) candidate reproduces the frozen dense basis;
2. its persistent transform cache scales linearly rather than quadratically;
3. it provides a substantial large-N crossover;
4. it does not improve the current short, truncated Plane bounded axis.

Therefore `dense` remains the production default. The safe next stage is an
R2R-C policy experiment that selects an implementation from explicit geometry,
axis size, retained fraction, dtype, device, and value type. Such a policy must
default to dense outside qualified cells, expose its decision in metadata, and
be tested on H100 before any production promotion. R2R-B by itself does not
authorize that promotion.
