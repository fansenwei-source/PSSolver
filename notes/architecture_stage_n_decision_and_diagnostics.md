# Architecture Stage N: retention decision and runtime diagnostics

## Decision

Retain the separated Plane architecture and its experimental shadow runtime,
but do not promote the shadow executor into production.  The current
`Plane_beris_edwards_stokes.py` remains the production implementation.  The
shadow remains an opt-in numerical oracle and migration target; Channel and
the generic solver remain outside this decision.

This is a decision about execution maturity, not mathematical validity.  The
Stage M H100 gate completed at commit
`21a8e95431cc1e6e971a5ed4a89f8326f185b68c`.  All 42 Q/u/p arrays were finite
and complete.  The maximum production/shadow relative-L2 error was
`3.758727291694351e-15`, against a fixed `1e-10` tolerance.  The separated
model, algebraic DAG, geometry dispatch, and timestep ordering therefore have
a qualified short-horizon numerical reference.

The same gate found that the shadow mean timestep was about 2.90 times the
production mean, while peak allocated memory was about 2.12 times production.
Those observations prohibit promotion but do not justify undoing the
responsibility separation established in Stages A--M.

## Diagnostic boundary

Stage N adds instrumentation, not an optimization.  Instrumentation is
created only when `enable_performance_instrumentation=True` is supplied to the
experimental runtime builder.  The default is `False`; in that state no CUDA
events, timers, profiler, synchronization points, or additional tensor
operations are introduced.

The opt-in semantic recorder measures nested regions for:

- the complete timestep;
- pre-RHS algebraic synchronization;
- every geometry-dispatched algebraic solve;
- physical materialization and publication of algebraic outputs;
- explicit model RHS and its projected publication;
- spectral update and dynamic projection;
- dynamic inverse transforms;
- scheduled spectral refreshes;
- model-context projected forward and inverse operations;
- every backend forward and inverse transform, including calls made directly
  by geometry-specific helpers.

CUDA timing uses deferred events.  Synchronization occurs only when a
diagnostic snapshot is explicitly requested.  CPU canaries use
`time.perf_counter`.  Regions are nested, so their totals must not be added
across hierarchy levels.

An independent, short PyTorch operator audit counts a frozen set of allocation
and movement operators: `empty`, `empty_like`, `empty_strided`, `clone`,
`copy_`, `to`, `_to_copy`, `contiguous`, `stack`, and `cat`.  It is deliberately
separate from semantic timing because profiler overhead would otherwise
contaminate the phase measurements.

## H100 diagnostic gate

The read-only Stage N plan reuses only the immutable Stage M production
reference.  It creates no new production trajectory and authorizes no
scientific benchmark.  One H100 job may run three independent shadow
diagnostics, each with ten warm-up steps, twenty semantic-timing steps, and a
separate two-step operator audit.  The resulting report ranks top-level
timestep regions and algebraic solvers and reports selected operator calls per
step.

A completed Stage N diagnostic makes the code eligible for a bounded Stage
N.1 optimization design.  It does not authorize an optimization, a default
change, production promotion, Channel migration, a long run, or any benchmark
claim.  Any optimization must subsequently preserve the Stage M numerical
gate and be compared against both the production reference and the qualified
pre-optimization shadow.
