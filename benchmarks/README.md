# PSSolver microbenchmarks

Microbenchmarks in this directory measure isolated numerical kernels. They are
not scientific validation tests and do not replace convergence or manufactured
solution tests.

Run the tensor-product transform benchmark from the repository root:

```bash
python -m benchmarks.benchmark_transform_backend \
  --device cuda \
  --dtype float64 \
  --shape 320,320,80 \
  --boundary-conditions periodic,periodic,neumann
```

The benchmark compares the active backend implementation with an explicit
legacy reference that applies one transform axis at a time. It checks numerical
agreement before reporting timings. CUDA measurements synchronize around each
timed region; setup and random input generation are excluded.

Measure the free-slip modal Stokes pressure-diagnostic overhead with:

~~~bash
python -m benchmarks.benchmark_stokes_pressure_diagnostics \
  --device cuda \
  --dtype float64 \
  --shape 320,320,80 \
  --lengths 100,100,20
~~~

This benchmark verifies that enabling or disabling residual measurements does
not change the computed velocity or pressure before reporting the speedup.

For comparisons between commits, use the same Python, PyTorch, hardware,
warm-up count, repeat count, shape, dtype, and boundary conditions. Record the
full JSON output together with the Git commit and machine identity.

Profile the complete Beris--Edwards--Stokes timestep with:

```bash
python -m benchmarks.profile_beris_edwards_timestep \
  --device cuda \
  --dtype float64 \
  --shape 128,128,32 \
  --lengths 100,100,20 \
  --dealias-rule cubic_half \
  --warmup-steps 3 \
  --profile-steps 10 \
  --output /tmp/pssolver_profile_128.json
```

The additive regions partition the numerical timestep. The nematic-force,
Stokes-solve, and transform regions are nested cross-cutting measurements and
must not be added to the partition. CUDA events are recorded asynchronously;
the benchmark synchronizes only after all profiled steps.

Snapshot transfer and I/O are disabled by default. To measure them separately,
provide both `--snapshot-interval` and a new or empty `--snapshot-directory`.
Add `--save-hydrodynamics` to include velocity and pressure files. Never use an
active scientific run directory as a benchmark snapshot destination.
