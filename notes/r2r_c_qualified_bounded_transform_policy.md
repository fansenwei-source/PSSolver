# R2R-C: qualified bounded-transform selection policy

## Decision

R2R-C adds an opt-in `auto` selector for bounded-axis DCT-II/DST-II
transforms. `auto` is not a heuristic and does not infer a crossover from the
axis length alone. It consumes an explicit qualification artifact and permits
the FFT implementation only when the complete runtime context matches an
allow-listed cell. Every unmatched context uses the dense reference.

The production default remains `dense`. R2R-C does not change the frozen
PSSolver v0.1 contract and does not promote the FFT implementation on any
machine.

## Selection identity

Each runtime decision records and checks:

- geometry and bounded local axis;
- DCT or DST and forward or inverse direction;
- full or truncated execution;
- physical size and retained mode count;
- transform-line count;
- device type and exact device name;
- real dtype and real/complex input type.

Qualification cells may require an exact local axis or deliberately apply to
all bounded axes of one geometry. They specify a minimum measured line count;
smaller batches fall back to dense. A cell can authorize only `fft`. The
fallback is fixed to `dense`, duplicate cell IDs are rejected, and two cells
matching the same runtime context are treated as an error rather than resolved
by ordering.

Selection is independent for forward and inverse transforms. This matters
because their crossover points need not coincide. The profiler records the
loaded policy, every observed context, the selected implementation, the
evidence identity, and the number of calls.

## Qualification artifact

`benchmarks/build_bounded_transform_policy.py` converts one frozen R2R-B
dense-versus-FFT JSON result into an allow-list. A direction is accepted only
when all of the following hold:

1. the source is a CUDA R2R-B artifact containing paired dense and FFT cases;
2. all candidate values are finite;
3. the maximum dense-reference relative L2 error satisfies the case tolerance;
4. the direction has the minimum requested timing-sample count;
5. median dense/FFT speedup satisfies the requested threshold;
6. candidate peak allocated memory satisfies the requested ratio limit.

The output retains the source SHA-256, environment, criteria, every accepted
or rejected direction, and an evidence ID in each selected cell. Policy files
are hardware-specific evidence, not portable performance promises.

## Local evidence

The local policy experiment used the R2R-B RTX 3060 Ti, float64, 4096-line
sweep with these conservative gates:

- median direction speedup at least 1.10;
- at least 10 timing samples;
- finite output and dense-reference error within the frozen case tolerance;
- peak allocated memory no more than three times the dense case.

It accepted 45 of 80 direction-specific cells. A dense/FFT/auto sweep over
N=40, 80, 160, and 320 confirmed that all automatic results remained within
2.704e-14 relative L2 of dense. For real full transforms, the observed
combined dense/auto speedups were:

| N | DCT | DST |
|---:|----:|----:|
| 40 | 1.286 | 1.168 |
| 80 | 1.400 | 1.222 |
| 160 | 1.782 | 1.566 |
| 320 | 3.655 | 3.206 |

For real half-retained transforms, the combined speedups were:

| N | DCT | DST |
|---:|----:|----:|
| 40 | 1.002 | 1.000 |
| 80 | 1.000 | 1.000 |
| 160 | 1.146 | 1.079 |
| 320 | 1.770 | 1.579 |

Values at or near one correspond to dense fallback or to a mixed
direction-by-direction choice. They must not be interpreted as universal
crossover thresholds.

## Complete Plane fallback check

A production-like local Plane comparison used shape 128 by 128 by 32,
float64, `cubic_half`, truncated projected transforms, `real_first`, Hermitian
half storage, and eager pointwise algebra. The local policy contained no
qualified physical-size-32 cells, so all 13 observed bounded-transform
contexts fell back to dense.

After eight measured steps:

- Q, velocity, and pressure were byte-for-byte identical to forced dense;
- the final-state SHA-256 was identical;
- peak allocated and reserved memory were identical;
- mean timestep was 40.652 ms for forced dense and 41.475 ms for `auto`.

The small `auto` overhead is expected when every call performs policy lookup
and metadata accounting. This check establishes safe fallback, not a reason to
enable `auto` for the current short Plane bounded axis.

## Interface and safety boundary

- `dense` remains the default and preserves the historical path.
- `fft` remains an explicit R2R-B override.
- `auto` requires a valid `QualifiedBoundedTransformPolicy`; it cannot run
  without one.
- Supplying a policy with a forced `dense` or `fft` selector is rejected.
- The complete-timestep profiler accepts `--bounded-transform-policy` only
  with `--bounded-transform-algorithm auto`.
- The standalone benchmark can exercise dense, FFT, and policy-selected cases
  from identical inputs.

R2R-C is complete as an experimental policy mechanism. Before a policy can be
used on H100, its R2R-B evidence must be regenerated on H100 and converted into
a separate H100-qualified artifact. Even a successful H100 artifact would
authorize only its exact cells; changing the production default requires a
separate decision and complete production-path qualification.
