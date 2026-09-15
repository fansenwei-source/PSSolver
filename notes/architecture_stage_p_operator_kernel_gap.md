# Architecture Stage P: production--canary operator/kernel gap

## Purpose

Stage O closed the Plane migration study with two deliberately different
roles. The optimized legacy runtime remains production; the separated runtime
remains a scientifically qualified numerical oracle and architecture
laboratory. Stage P does not reopen that decision. It asks a narrower question:
which operator, kernel-launch, transform, or compiled-graph signals explain the
remaining production--canary timestep gap at the R320 benchmark scale?

This is a diagnostic stage, not an optimization stage. It changes no equation,
boundary condition, tensor layout, field lifetime, runtime implementation, or
default. It does not create a long trajectory or a new benchmark result.

## Measurement contract

Each runtime is reconstructed independently from the same completed R320
production reference. Configuration metadata and the exact `Q_0.npy` identity
are checked before and after every profile. Both roles use the frozen
production numerical choices: float64, cubic-half dealiasing, truncated
projected transforms, Hermitian-half storage, real-first execution, compiled
pointwise kernels, spectral molecular field and stress divergence, and the
same spectral-refresh interval.

Three balanced trials are required per runtime. Every trial contains three
fresh-runtime windows:

1. **Authoritative throughput:** ten warm-up steps and twenty timed steps,
   using only an outer pair of CUDA events. Internal semantic instrumentation
   is disabled for both runtimes.
2. **Operator/kernel audit:** a separate ten-step warm-up followed by three
   steps under the PyTorch profiler. Only bounded aggregates are retained:
   complete operator names, complete kernel names, calls/launches per step,
   device time per step, CUDA-runtime entries, compiled-region markers, and
   coarse categories. No Chrome trace or raw profiler event is written.
3. **Semantic attribution:** another fresh runtime, ten warm-up steps, and
   three instrumented steps. Matched transform, algebraic, explicit-RHS,
   spectral-update, dynamic-inverse, refresh, and total regions are reported.
   These regions are nested and are not authoritative throughput.

The separation is essential. CUDA-event instrumentation creates events and
Python bookkeeping of its own, and nested regions may overlap. The earlier
Stage O.4.1 production--canary ratio therefore remains historical evidence,
but Stage P does not assume that its entire measured gap was intrinsic to the
architecture. It first remeasures both paths under matched, non-instrumented
throughput conditions.

## Outputs and interpretation

`benchmarks/profile_plane_stage_p.py` writes one bounded JSON profile for one
runtime. `scripts_plane/analyze_plane_stage_p.py` accepts exactly three
distinct profiles per runtime, verifies their common closure/configuration/Q
identity, and reports:

- the authoritative canary/production timestep ratio and absolute gap;
- kernel launches, CUDA runtime calls, and compiled markers per step;
- matched transform-region timing from the separate semantic window;
- complete per-name operator and kernel deltas, ranked by positive device-time
  contribution.

Kernel durations may overlap and profiler timings include observer overhead,
so no single row is automatically an optimization target. A successful Stage
P result is `DIAGNOSTIC_COMPLETE` and authorizes only a human review of a
possible Stage Q target. It explicitly does not authorize a Stage Q candidate,
production promotion, default change, or renewed projected-materialization
work.

## Execution boundary

`scripts_plane/plan_plane_stage_p.py` creates a planning-only six-profile
balanced command list and one analysis command. The plan itself creates no
directory, solver, CUDA context, or output. Executing the plan requires a new,
explicitly authorized single H100 job after CPU tests pass. Any identity drift,
graph break, non-finite state, incomplete aggregate, raw-trace retention, or
nonzero command exit stops the stage without retry.

Production Plane, Channel, the generic solver, and benchmark `develop` remain
outside all Stage P imports.
