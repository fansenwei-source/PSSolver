# Phase 4 plan: second integrator and bounded modal block operators

Status: `P4_3_LOCAL_COMPLETE_REFERENCE_ONLY`.

Baseline: Phase 3 closure record commit
`e1fcc5d6df5ea6cad6b61cc770a85d19c04f9953` on
`next/pssolver-v0.2.0-architecture`.

Phase 4 is explicitly authorized after Phase 3 closure.  This authorization
starts the phase; it does not change a production default, promote
`separated_canary`, connect a new integrator to Plane, or authorize Phase 5.

## Objective

Prove that the v0.2 state/workspace/step-program architecture supports:

1. a second, genuinely multistep time integrator; and
2. a small coupled implicit spectral operator rather than only scalar
   denominators.

The chosen integrator is constant-step SBDF2.  The qualifying coupled problem
is a two-component periodic reaction--diffusion system with an analytic modal
solution.  Semi-implicit Euler remains the comparison and startup method.

## Numerical contract

For `du/dt = L u + N(u)`, startup advances one successful step with the
qualified projected semi-implicit Euler program.  Later steps use

```text
(3 u[n+1] - 4 u[n] + u[n-1]) / (2 dt)
    = L u[n+1] + 2 N(u[n]) - N(u[n-1]).
```

The order of one successful SBDF2 step is:

1. require current state and complete compatible history;
2. prepare required algebraic values;
3. invoke the pre-update callback;
4. evaluate the current explicit RHS;
5. assemble the SBDF2 implicit right-hand side;
6. solve the scalar diagonal or qualified 2-by-2 modal block operator;
7. project evolved spectra;
8. inverse-transform evolved fields;
9. perform a scheduled spectral refresh when due;
10. commit progress;
11. commit the previous-state and previous-RHS history last.

Any exception aborts the workspace generation and preserves the last complete
state, progress, representation ledger, and integrator history.

## Ownership

Persistent runtime state gains integrator-owned history with explicit scheme
identity and history depth.  For SBDF2 the persistent tensors are:

- previous evolved native spectrum;
- previous explicit native-spectral RHS.

Current RHS, assembled solve RHS, block-solve scratch, and temporary
reconstruction buffers remain bounded workspace.  Linear coefficients,
component order, modal matrices, and selected solve implementations belong to
immutable plans.  Diagnostics may summarize history but never retain mutable
history tensor references.

## Basic modal block scope

The first qualified block operator has exactly two coupled components.  It is
constant-coefficient in physical space and independent across retained
spectral modes.  Its bound coefficient tensor has trailing shape `(2, 2)` and
its RHS has trailing component size `2` after a single documented packing
convention is selected.

Binding must reject:

- wrong component order or shape;
- real/spectral dtype incompatibility;
- device mismatch;
- NaN or Inf coefficients;
- singular or numerically rejected modal matrices;
- an unsupported block size;
- a geometry or basis capability mismatch.

The implementation may cache a factorization but may not recompute registry
dispatch or allocate unbounded temporary tensors inside each timestep.

## Checkpoint policy

Plane workflow checkpoint format v1 remains byte-for-byte unchanged.  The
Phase 4 reference workflow uses a separate provisional generic checkpoint
schema containing:

- integrator identity and schema version;
- `dt`, completed steps, and startup/multistep phase;
- current evolved state and representation validity;
- previous evolved spectrum and previous explicit RHS;
- tensor shape, dtype, finite status, and SHA-256 records.

Missing history, tampered history, a changed `dt`, a different integrator, an
unsupported format version, or incompatible model/discretization identity is
rejected before advancing or writing output.

## Slices

### P4.0: design and oracle freeze

Add this plan, ADR 0010, a machine-readable contract, static tests, source
identity, scope exclusions, and qualification thresholds.  Numerical code is
unchanged.

### P4.1: tensor-free integrator declaration and disconnected history

Introduce a tensor-free integrator specification and disconnected persistent
history contract.  Do not connect either object to Plane or a production
runtime.

Completed locally.  The result is recorded in
[phase_4_p41_integrator_history.md](phase_4_p41_integrator_history.md) and its
[machine-readable evidence](phase_4_p41_integrator_history.json).  The new
types remain direct-import-only and disconnected from every runtime.

### P4.2: scalar SBDF2 reference path

Implement SBDF2 over a scalar periodic diffusion/reaction reference model.
Freeze startup, callback, refresh, failure atomicity, and metadata behavior.

Completed locally.  The result is recorded in
[phase_4_p42_scalar_sbdf2_reference.md](phase_4_p42_scalar_sbdf2_reference.md)
and its [machine-readable evidence](phase_4_p42_scalar_sbdf2_reference.json).
The implementation is a CPU float64 functional reference and remains
disconnected from every production runtime.

### P4.3: convergence and restart

Verify Euler order at least `0.9` and SBDF2 order at least `1.8` over a
pre-registered halving ladder.  Add continuous/split equivalence and negative
history/checkpoint gates.

Completed locally.  The convergence results, provisional reference checkpoint,
byte-identical split/restart result, and negative gates are recorded in
[phase_4_p43_convergence_restart.md](phase_4_p43_convergence_restart.md) and
its [machine-readable evidence](phase_4_p43_convergence_restart.json).  Plane
checkpoint v1 and all production runtimes remain unchanged.

### P4.4: two-component modal block operator

Add tensor-free coupling declarations, bound 2-by-2 modal coefficients, a CPU
reference solve, an optimized GPU solve, and a two-component analytic
reaction--diffusion qualification model.

### P4.5: combined SBDF2/block canary

Run the block-coupled reference model through SBDF2 with bounded workspaces,
fixed history ownership, restart, and metadata provenance.  It remains an
opt-in reference path.

### P4.6: closure qualification

Run the complete CPU suite, wheel/source import checks, analytic convergence,
continuous/restart comparisons, negative gates, and balanced H100 performance,
memory, finite-value, fallback, and allocation checks.  A separate decision is
required before Phase 5.

## Qualification oracle

- temporal grids use the fixed ratio ladder `(1, 1/2, 1/4, 1/8)` at unchanged
  spatial resolution and final physical time;
- Euler observed L2 order must be at least `0.9`;
- SBDF2 observed L2 order must be at least `1.8`;
- the analytic model must also report Linf error and finite status;
- same-backend split/restart final state and history must be byte-identical to
  continuous execution when the same operation sequence is used;
- CPU and GPU results use pre-registered float64 mixed tolerances when their
  operation implementations differ;
- peak allocated and reserved memory must remain within a pre-registered bound
  for each compared implementation;
- no implicit fallback, graph break, NaN, Inf, OOM, or CUDA error is allowed.

## Explicit exclusions

Phase 4 does not implement or authorize:

- CNAB2 or more integration schemes;
- adaptive or variable timesteps;
- arbitrary-size block algebra;
- variable-coefficient implicit operators;
- symbolic equation construction;
- Newton, Krylov, PETSc, MPI, or multi-GPU execution;
- Plane Beris--Edwards migration;
- a new geometry or physical boundary law;
- strong anchoring or optimal control;
- production runtime or default promotion.

The machine-readable authority is
[phase_4_contract.json](phase_4_contract.json).
