# Truncated projected transforms with Hermitian half-spectrum storage

## Scope

This branch combines two independently controlled transform optimizations on
top of commit `fe7bf866994c77a18351c80ad805ff39b5727b1e`:

- `projected_transform_execution={full,truncated}` controls whether projected
  DCT/DST axes evaluate their complete matrices or only retained real-basis
  modes.
- `spectral_storage={full_complex,hermitian_half}` controls whether the second
  periodic Plane axis uses a complete complex FFT or native real-input
  Hermitian storage.

The production default remains `truncated + full_complex`.  Half-spectrum
storage remains explicit and opt-in until the combined H100 gate passes.

The integration extends retained-axis transforms through the rFFT backend:
periodic axes retain the backend's complete native spectral extent, while
DCT/DST axes may use reduced matrices.  Inverse transforms validate and crop
the backend spectral shape before `irfftn`, then expand retained real-basis
modes back to the physical grid.  No basis, normalization, PDE, boundary
condition, or dealiasing cutoff changes.

## Local correctness gates

The combined worktree passed 171 focused transform, profiler, CLI, dtype, and
manufactured-solution tests.  After the final metadata clarification, the
complete local suite passed 614 tests and eight subtests.

A 32x32x16 float64 CUDA production smoke ran 100 steps for all four controls.
`full` and `truncated` were byte-identical within each storage backend.  The
Hermitian paths differed from the full-complex paths only at roundoff:

| field | relative L2 | Linf |
| --- | ---: | ---: |
| Q | 4.525e-16 | 3.886e-16 |
| u | 4.572e-16 | 2.949e-17 |
| p | 4.932e-16 | 3.903e-18 |

All four runs completed with finite Q, u, and p outputs.

## Local performance signal

An RTX 3060 Ti transform-focused eager-pointwise profile used a 64x64x32
float64 grid, three warmup steps, and ten timed steps:

| ID | projected transform | spectral storage | ms/step | peak allocated MiB | peak reserved MiB |
| --- | --- | --- | ---: | ---: | ---: |
| A | full | full_complex | 17.227 | 149.68 | 226 |
| B | truncated | full_complex | 13.900 | 146.08 | 212 |
| C | full | hermitian_half | 13.389 | 103.27 | 168 |
| D | truncated | hermitian_half | 11.090 | 105.39 | 168 |

Observed speedups were A/B `1.239x`, A/C `1.287x`, A/D `1.553x`, B/D
`1.253x`, and C/D `1.207x`.  These desktop results establish a useful signal,
not a production qualification; the final decision requires a balanced H100
2x2 test with compiled pointwise kernels at 320x320x80.

## Promotion boundary

This branch does not change `DEFAULT_SPECTRAL_STORAGE`; it remains
`full_complex`.  A future default promotion requires:

1. zero CPU regressions;
2. four complete H100 profiler paths with explicit metadata;
3. 100-step production equivalence below the established `1e-12` relative-L2
   tolerance for half-spectrum comparisons;
4. stable R320 speed and memory improvement for the combined path;
5. no compile fallback, graph break, nonfinite value, CUDA error, or OOM.
