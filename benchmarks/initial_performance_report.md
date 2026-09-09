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
