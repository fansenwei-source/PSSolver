# Periodic fast-path candidate report

## Scope

This post-v0.1 branch tests two independently selectable optimizations inspired
by the pinned upstream `aaveg/PSSolver` implementation at commit
`f2660d309b266d827df06fdc10ad59d2f6ea9b08`:

1. replace Python-list advanced indexing with a contiguous slice when a
   transform group occupies adjacent field slots;
2. replace multiple periodic-axis `fft`/`ifft` calls with one
   multidimensional `fftn`/`ifftn` call.

Both selectors are opt-in. The frozen v0.1 defaults remain `advanced` and
`axiswise`. Noncontiguous field groups and mixed-boundary legacy execution
fall back to the historical paths. Hermitian-half Plane transforms remain on
their existing `rfftn`/`irfftn` implementation.

## Local qualification

Environment:

- NVIDIA GeForce RTX 3060 Ti;
- PyTorch 2.6.0, CUDA runtime 12.6;
- four dynamic fields, batch size one;
- complete semi-implicit periodic diffusion timesteps;
- five warm-up steps, 30 measured steps, four balanced trials;
- float32 unless noted otherwise.

The variants are:

- A: advanced indexing + axiswise periodic FFT;
- B: contiguous slice + axiswise periodic FFT;
- C: advanced indexing + multidimensional periodic FFT;
- D: contiguous slice + multidimensional periodic FFT.

| Grid | A (ms) | B (ms) | C (ms) | D (ms) | A/D speedup | A/D peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x128 | 0.067288 | 0.041252 | 0.054657 | 0.032840 | 2.049x | 4.439 / 3.439 MiB |
| 256x256 | 0.124595 | 0.091706 | 0.098256 | 0.067436 | 1.848x | 17.754 / 13.754 MiB |
| 512x512 | 0.510819 | 0.447807 | 0.414397 | 0.352008 | 1.451x | 71.008 / 55.008 MiB |
| 1024x1024 | 1.888492 | 1.702338 | 1.546158 | 1.356079 | 1.393x | 284.016 / 220.016 MiB |

The maximum float32 relative L2 difference over the four sweeps was
`3.393e-7`. A separate 512x512 float64 run produced:

- A = 1.538640 ms;
- B = 1.430981 ms;
- C = 1.381315 ms;
- D = 1.278829 ms;
- A/D = 1.203x;
- maximum relative L2 = `5.327e-16`;
- A/D peak allocated = 142.016 / 110.016 MiB.

The complete local CPU suite passed with `1040 passed, 8 subtests passed`.

## Decision

The candidate is locally promising and explains a substantial part of the
periodic benchmark gap, but this report does not promote either default. A
separate H100 qualification should repeat the balanced A/B/C/D experiment,
check float64 complete trajectories, and confirm that Plane's existing
Hermitian-half production path is unchanged before any default decision.
