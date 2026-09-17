# R2R-C: qualified bounded-transform selection policy

## Decision

R2R-C adds an opt-in `auto` selector for bounded-axis DCT-II/DST-II
transforms. `auto` is not a heuristic and does not infer a crossover from the
axis length alone. It consumes an explicit qualification artifact and permits
the FFT implementation only when the complete runtime context matches an
allow-listed cell. Every unmatched context uses the dense reference.

The production default remains `dense`. The completed H100 qualification
classified R2R-C as `C_rejected`: the mechanism is numerically correct and
fails safely, but it selected no FFT cell and its dispatch/accounting overhead
regressed the small Plane case. R2R-C therefore does not change the frozen
PSSolver v0.1 contract and is closed without a production promotion.

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

## H100 qualification and closure

The authoritative H100 run used commit
`ed4efcf639d8dc93a8b28679cc2755464d483275` in Slurm Job `10833253` on an
NVIDIA H100 PCIe. The job completed normally in 4 minutes 23 seconds, with one
submission and no retry. The CPU gates passed with 156 focused tests and 1,134
full-suite tests; the only skip was the pre-existing optional `nematics3d`
dependency test.

The frozen H100 R2R-B sweep reviewed 96 direction-specific cases using a
minimum median speedup of 1.10 and a maximum FFT/dense peak-allocated-memory
ratio of 1.05. No case passed the complete qualification contract:

- reviewed directions: 96;
- accepted cells: 0;
- rejected directions: 96.

Consequently both Plane resolutions used dense fallback for all 13 observed
contexts and selected zero qualified FFT contexts. This exercised the intended
fail-closed behavior. At both R128 and R320, final Q, velocity, and pressure
were byte-for-byte identical to forced dense; all reported relative L2 errors
were exactly zero.

The complete-timestep performance result was:

| resolution | dense mean (ms) | auto mean (ms) | auto/dense mean | result |
|---|---:|---:|---:|---|
| R128 | 7.181127 | 7.794546 | 1.085421 | 8.54% regression |
| R320 | 48.954697 | 48.987508 | 1.000670 | neutral |

At R128 the median auto/dense ratio was 1.088788. This exceeded the frozen 5%
non-regression limit, so the formal result was `C_rejected`. R320 was neutral,
but it selected no FFT cell and provided no speedup. There were no OOMs,
non-finite values, CUDA failures, compile fallbacks, graph breaks, or policy
ambiguities.

The authoritative control archive is
`/home/fansenwei/pssolver_r2r_c_h100_ed4efcf_20260917_v1`; its checksum
manifest verified 112 of 112 entries and has SHA-256
`c5e27bfcc71ec683e66e8e26dd8e73ae59cd23eae2be3080ff8fca47e3aea3f2`.
Large trajectory arrays remain under the corresponding `/scratch1` result
root recorded by that archive.

The rejection is a performance result, not a numerical or spectral-method
failure. It establishes that an audited per-call `auto` selector is not useful
for the current H100 Plane workload when no bounded transform meets the strict
speedup and memory gates. The explicit R2R-B FFT implementation remains a
research option for substantially larger bounded axes or different hardware,
but the current Plane line is closed.

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

R2R-C is complete as an experimental policy mechanism and closed as a
production candidate for the current H100 Plane workload. No default-promotion
smoke is authorized. Any future reconsideration requires a materially new
workload or implementation, fresh hardware-specific evidence, and a separate
production-path qualification; this rejected policy must not be reused as if
it were a qualified accelerator-wide crossover table.
