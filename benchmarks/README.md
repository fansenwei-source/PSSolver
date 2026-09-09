# PSSolver microbenchmarks

Microbenchmarks in this directory measure isolated numerical kernels. They are
not scientific validation tests and do not replace convergence or manufactured
solution tests.

Run the tensor-product transform benchmark from the repository root:

```bash
python benchmarks/benchmark_transform_backend.py \
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
