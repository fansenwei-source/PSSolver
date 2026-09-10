# Hermitian half-spectrum candidate

Date: 2026-09-10

Base commit: `9a67155b03f4863fa7088dd26721889e7ce3cbb4`

Branch: `perf/hermitian-half-spectrum-9a67155`

## Scope

This candidate adds an opt-in `hermitian_half` native spectral-storage mode
for real Plane fields. It packs the positive-y Fourier half-spectrum while
retaining the existing full x Fourier axis and z DCT/DST bases. The physical
grid, basis functions, normalization, differential operators, dealiasing
cutoffs, IMEX method, and saved Q/u/p snapshot formats are unchanged.

The production default remains `full_complex`. The candidate is exposed only
through the Beris--Edwards Plane driver and its profiling tools. Channel is not
changed and cannot select this Plane-specific configuration implicitly.

## Safety and numerical checks

- Hermitian storage requires `real_first` and an explicitly selected periodic
  packed axis. Unsupported boundary conditions fail rather than falling back.
- Even-grid first-derivative Nyquist coefficients are set to zero explicitly,
  matching the real collocation-grid result of the full-complex inverse.
- The PDE field container, linear IMEX operator, nonlinear output, static
  output, projector mask, wavevectors, and free-slip Stokes solver all use the
  backend's native spectral shape.
- The default full-complex 100-step Q/u/p snapshots are byte-for-byte identical
  to the base commit.
- A 100-step float64 full-versus-half comparison at `64x64x32`, including a
  spectral refresh every 20 steps, produced relative L2 differences of
  `4.41e-16` for Q, `2.28e-15` for u, and `1.14e-14` for p.
- The complete local CPU suite passed: `543 passed, 8 subtests passed`.

## Local GPU pilot

Hardware: NVIDIA GeForce RTX 3060 Ti, 8 GiB. PyTorch 2.6.0, CUDA runtime 12.6.
All measurements used float64, eager pointwise kernels, cubic-half dealiasing,
five warm-up steps, twenty measured steps, and three balanced-order trials.

| Grid | Storage | Mean step | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| `64x64x32` | full complex | 17.237 ms | 0.146 GiB | 0.221 GiB |
| `64x64x32` | Hermitian half | 13.770 ms | 0.101 GiB | 0.164 GiB |
| `128x128x32` | full complex | 69.410 ms | 0.549 GiB | 0.848 GiB |
| `128x128x32` | Hermitian half | 52.509 ms | 0.373 GiB | 0.633 GiB |

The mean local speedups were `1.252x` and `1.322x`, respectively. At
`128x128x32`, the peak allocated-memory ratio was approximately `0.679`.

The local TorchInductor path was not qualified because this desktop's Triton
toolchain could not find `cuda.h`. The candidate does not permit a silent
compile-to-eager fallback. Production-compile timing and the `320x320x80`
memory result therefore remain H100 qualification work.

## Promotion rule

Do not change the default from `full_complex` unless an isolated H100 A/B/C
job passes the complete CPU suite, compiled production-path performance and
memory gates, and a bounded float64 Q/u/p trajectory comparison. A failed or
neutral H100 result leaves this branch as an experimental opt-in backend.
