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

The complete-timestep profiler enables single-step Q-gradient reuse by
default. Pass `--disable-q-gradient-reuse` to measure the otherwise identical
uncached path with the same initial condition and numerical parameters.

## Nsight Systems capture

`capture_beris_edwards_nsys.py` reuses the same complete-timestep workload,
adds nested NVTX ranges, and brackets only the requested timesteps with the
CUDA profiler API. Run it through Nsight Systems, for example:

```bash
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output=/tmp/pssolver_nsys_64 \
  python -m benchmarks.capture_beris_edwards_nsys \
  --shape 64,64,32 --warmup-steps 5 --capture-steps 3 \
  --output /tmp/pssolver_nsys_64.json
```

Keep the `.nsys-rep` and any exported tables outside scientific run
directories. The capture runner does not save Q, velocity, or pressure arrays.

## Stage O.4.3 packed-publication qualification

`profile_plane_stage_o43.py` compares the frozen separated-canary storage path
with an opt-in preallocated algebraic-publication layout. The candidate lets
the boundary scheduler replace only provably contiguous `torch.cat` batches
with zero-copy views; every other batch uses the historical copy fallback.

Generate the exact bounded H100 command contract with
`scripts_plane/plan_plane_stage_o43_qualification.py`. The plan requires the
closed Stage O.4.2 accounting report, immutable R128/R320 production
references, a 100-step numerical comparison, and three balanced profile trials
per role and resolution. An `A_recommended` result authorizes only a later
architecture decision and never changes a production default.

## Stage O.4.3.1 natural-storage-view qualification

Stage O.4.3.1 keeps the historical `deferred_stack` publication policy on both
sides of the comparison. Only the candidate enables conservative contiguous
storage views for transform batches, so it can reuse adjacency already created
by an upstream producer without allocating or copying into the rejected
generation-wide packed store.

Generate the bounded H100 contract with
`scripts_plane/plan_plane_stage_o431_qualification.py`. The plan binds both
the closed Stage O.4.2 accounting report and the Stage O.4.3 rejection report,
then repeats the 100-step numerical gate and balanced R128/R320 profiles. No
result from this stage changes a production default.

## Stage O.4.3.3 producer-owned H/stress packing

Stage O.4.3.3 keeps the O.4.3.1 safe-view scheduler on both sides and changes
only the candidate molecular-field and stress producers.  The candidate emits
component-major tensors in Plane boundary-compatible storage order from inside
the pointwise kernel; the scheduler consumes those tensors through zero-copy
views.  Q RHS and gradient producers are deliberately outside this stage.

Generate the bounded H100 contract with
`scripts_plane/plan_plane_stage_o433_qualification.py`.  The plan binds the
Stage O.4.2 accounting result, the rejected O.4.3 scheduler-republication
result, and the performance-neutral O.4.3.1 natural-view result.  It runs one
100-step numerical comparison plus three balanced R128/R320 profile trials per
role.  No result changes a production default.

## Stage O.4.3.4 materialization attribution

Stage O.4.3.4 keeps the neutral O.4.3.3 candidate unchanged and attributes
every projected-transform batch assembly to its algebraic or explicit-RHS
caller. `profile_plane_stage_o434.py` records per-source batch counts, logical
bytes, exact copy/cat output bytes, contiguous views, fallback reasons, and
deferred-event timing without retaining tensors.

Generate the diagnostic-only R320 command contract with
`scripts_plane/plan_plane_stage_o434_diagnostic.py`. The plan binds the frozen
O.4.3.3 report, runs three twenty-step attribution profiles, and ranks the
remaining copy/cat sources. Its result can guide target review only; it cannot
authorize another layout, Stage O.4.4, or a production default change.

## Stage O closure

`scripts_plane/close_plane_stage_o.py` emits the final read-only Stage O
architecture decision. It binds the authoritative O.4 through O.4.3.4 report
identities and records that legacy Plane production remains the default, the
separated runtime remains an experimental numerical oracle, and the projected
materialization optimization line is closed. The command constructs no solver
and authorizes only a local Stage P operator/kernel diagnostic design.

## Stage P operator/kernel gap diagnostic

`profile_plane_stage_p.py` compares the retained Plane production runtime with
the separated numerical oracle at R320 without changing either implementation.
It deliberately isolates non-instrumented throughput, bounded PyTorch
operator/kernel aggregation, and nested semantic timing in three fresh-runtime
windows. It never exports or retains a raw profiler trace.

Generate the balanced three-trial-per-runtime contract with
`scripts_plane/plan_plane_stage_p.py`, then aggregate it with
`scripts_plane/analyze_plane_stage_p.py`. A complete result permits only Stage
Q target review. It does not select an optimization, reopen projected
materialization work, or authorize a production/default change.
