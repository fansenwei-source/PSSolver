# ADR 0013: functional runtime and external control ownership

Status: accepted for Phase 9.

## Context

Phase 8 closed a tensor-free public `Simulation` declaration and four
qualified model--geometry applications. PSSolver also still contains the
older `pssolver.control` research package. That package proved that one-step
autograd and a checkpointed discrete adjoint were useful, but its functional
adapter replaces private field buffers, writes the model parameter registry,
clears a pressure warm start, and reimplements the time-integrator update.
Its objectives, actuator parameterizations, optimization algorithms, and loop
diagnostics are application research rather than PDE-solver responsibilities.

An independent control consumer now needs a differentiable, deterministic,
and provenance-bearing PDE execution contract. Copying the consumer back into
PSSolver would reverse the desired dependency direction and couple the solver
to one optimization method. Adapting the mutable legacy solver from outside
would make private storage layout and hidden warm state part of an accidental
API.

## Decision

PSSolver will own a provisional functional runtime. It exposes an immutable
flat tuple of tensors with a declared layout, a pure one-step transition,
named differentiable observations, explicit field-control injection,
scientific/discretization/execution identity, deterministic replay
capabilities, and durable checkpoint conversion. The functional and
production executors may differ in allocation, fusion, and compilation, but
they implement the same declared discrete scientific map.

The first vertical slice is complete-stress Beris--Edwards in a fully periodic
box with a spatial activity field. The periodic direct Fourier Stokes solve
avoids pressure PCG, lifting, Robin walls, and bounded-domain null modes. A
Channel functional runtime and its pressure-solve differentiation follow only
after the periodic slice is validated.

Observations returned by `step_and_observe(state, control, step_index)` belong
to the input state under that step's control. They must be bitwise identical
to a separate `observe(state, control)` call while allowing the algebraic flow
solve to be shared. Terminal observation without a control omits
control-dependent algebraic fields.

Functional replay is bitwise under one device, dtype, and execution identity.
Production-versus-functional validation instead uses strict predeclared
tolerances because the two executors may legally order equivalent floating-
point operations differently. Unsupported capabilities fail at construction;
there is no runtime fallback.

The state contract always represents a batch dimension. The first slice may
qualify batch size one and declare larger batches unsupported. The concrete
periodic state packing is selected in P9.1 and becomes part of the runtime
identity; P9.0 does not freeze a permanent package-wide field packing.

PSSolver owns model equations, admissible physical control ranges, the exact
equation injection term, state and observation conventions, functional
execution, deterministic replay declarations, and durable restart formats.
The independent consumer owns actuator bases, temporal parameterization,
optimization bounds within the physical range, objectives, checkpoint
scheduling for an adjoint evaluation, adjoint orchestration, optimizers, and
experiments. The dependency direction is consumer to PSSolver only.

The legacy `pssolver.control` package is frozen as compatibility-public oracle
code. Its import paths and behavior remain available; it receives only
compatibility-preserving bug fixes. It is not extended into the new API and is
not removed until consumer migration and the ADR 0005 deprecation window are
complete.

## Consequences

- Phase 9 starts with a periodic activity-control slice rather than Channel.
- The periodic positive-friction branch and z-invariant execution are
  prerequisites to be publicly qualified, not new control algorithms.
- The control field enters active force as
  `div(beta * alpha * Q)`: the product is formed before differentiation and
  its dealiasing rule is part of the discretization identity.
- Pressure warm starts and any other persistent algebraic state must either
  appear in the functional state or be disabled in functional execution.
- Diagnostic caches cannot affect a later functional step.
- Checkpoint scheduling and Revolve-like algorithms do not enter PSSolver in
  this phase; PSSolver supplies complete state copy/serialization and size
  metadata.
- Batch sizes greater than one, custom VJP/JVP implementations, and implicit
  pressure adjoints are capability extensions rather than prerequisites for
  the first batch-one periodic reference.
- The functional API remains provisional until a periodic consumer and a
  Channel consumer both qualify it.

## Qualification boundary

P9.0 records ownership, requirements, oracle identity, and slice order only.
It changes no runtime, timestep, model equation, numerical operator,
checkpoint schema, output schema, public package-root export, or production
default. P9.1 and every H100 submission require separate authorization.
