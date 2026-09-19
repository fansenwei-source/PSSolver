# Phase 1 plan: mechanical extraction of `transforms.py`

Status: planned, not yet authorized by this Phase 0 commit.

## Objective

Split the 1,631-line `pssolver/transforms.py` responsibility cluster into
canonical numerical modules while preserving the existing import facade,
tensor operations, defaults, Plane runtime, and numerical results.

Phase 1 does not redesign `Fields`, configuration, state ownership, the
timestep, boundary semantics, or transform algorithms.

## Target ownership

```text
pssolver/backends/tensor_product.py
    dtype helpers
    transform-order, spectral-storage, and periodic-execution constants
    TransformMetadata
    TensorProductTransformBackend

pssolver/operators/projection.py
    dealias and projected-execution constants
    BasisAwareSpectralProjector
    projected forward/inverse helpers

pssolver/operators/tensor_divergence.py
    projected_common_basis_stress_divergence
    projected_distortion_stress_divergence
    their private spectral-gradient helpers

pssolver/linear_solvers/stokes/plane_free_slip.py
    FreeSlipModalStokesSolver

pssolver/transforms.py
    compatibility imports and the historical public names
```

The second divergence helper contains a qualified three-dimensional parity
split and retains its historical distortion-stress name for compatibility.
Its implementation consumes only a row-major tensor, explicit basis
signatures, and a backend; it does not construct active-nematic stress.  It is
therefore recorded as a specialized tensor operator rather than as generic
physics.  If Phase 1 characterization finds hidden model assumptions, the
helper stays in a clearly named compatibility module instead of being
misrepresented as a universal operator.

## Prospective commit sequence

### P1.0: characterize the facade

- freeze the v0.1.2 `pssolver.transforms` public symbol inventory;
- verify root re-exports and constructor signatures;
- verify implementation provenance and checkpoint data do not pickle concrete
  classes;
- capture a short legacy Plane Q/u/p identity control.

### P1.1: create destination packages

- add `pssolver/operators`;
- add `pssolver/linear_solvers/stokes`;
- add package `__init__` files with no default or runtime change.

### P1.2: extract the tensor-product backend

Move dtype helpers, transform-order/storage/periodic-execution constants,
`TransformMetadata`, and `TensorProductTransformBackend` to
`pssolver/backends/tensor_product.py`.
`pssolver.transforms` immediately re-exports every historical public symbol.

This must happen first so later operator and solver modules can depend on the
canonical backend without importing the compatibility facade and creating a
cycle.

### P1.3: extract spectral projection

Move dealias/projected-execution constants, `BasisAwareSpectralProjector`, and
its private projected-transform helpers to `pssolver/operators/projection.py`.
The semantic dealias enum remains owned by `core.numerics`; this move only
relocates the v0.1.2 execution constants without redesigning policy.  Preserve
method order, signatures, defaults, allocation behavior, and operation order.

### P1.4: extract stress divergence

Move the two projected tensor-divergence functions and their private gradient
helpers to `pssolver/operators/tensor_divergence.py`.

Stress construction remains model-owned.  Generic tensor divergence becomes
operator-owned.  This commit changes ownership only; it does not rewrite the
functions into a new abstraction.

### P1.5: extract Plane free-slip Stokes

Move `FreeSlipModalStokesSolver` to
`pssolver/linear_solvers/stokes/plane_free_slip.py`.  Keep all pressure,
zero-mode, parity, diagnostic, and warm-start behavior unchanged.

### P1.6: reduce `transforms.py` to a compatibility facade

- import and re-export the historical names;
- keep their public import paths working;
- add an explicit compatibility inventory;
- migrate repository-internal imports to canonical modules in separate,
  reviewable changes;
- do not remove the facade in v0.2.

## Frozen non-changes

Every extraction commit preserves:

- function and constructor signatures;
- constant names and values;
- transform normalization and mode order;
- bounded and periodic execution order;
- full/half spectral storage semantics;
- projected-transform truncation semantics;
- tensor layout, batching, and workspace behavior;
- operation order and floating-point results;
- transform-call counts;
- Plane equations, boundary conditions, pressure gauge, and zero-mode policy;
- runtime selector and omitted-selector behavior;
- CLI, output, restart, and metadata scientific meaning.

The concrete class `__module__` will change when implementation ownership
moves.  Before extraction, tests must confirm that checkpoints do not pickle
these classes.  Provenance may record the new canonical source module as an
execution-identity change; it must not be represented as a scientific change.

## Per-commit local gates

- facade/import characterization tests;
- transform reference, odd/even, full/truncated, and full/half-storage tests;
- manufactured free-slip Stokes and pressure diagnostics;
- dependency-boundary tests;
- complete CPU suite;
- short legacy Plane Q/u/p byte identity;
- `git diff --check`;
- clean wheel import check when package layout changes.

No HPCC job is required for each mechanical move.

## Phase-completion gates

After the complete extraction, run one balanced H100 non-regression task:

- R128 and R320;
- three balanced parent/candidate pairs;
- candidate mean timestep ratio at most `1.02` for a pure extraction;
- peak allocated and reserved ratios at most `1.02`;
- identical forward/inverse transform counts;
- no graph break, compile fallback, transform fallback, OOM, NaN, Inf, or CUDA
  error;
- a 100-step production trajectory with byte-identical Q/u/p;
- restart and output-schema checks.

Failure retains v0.1.2 unchanged and stops Phase 1.  Passing Phase 1 does not
change a production default because only module ownership has changed.

## Explicitly deferred

- `Fields`/state ownership redesign;
- RunSpec decomposition;
- compiled generic runtime;
- new DCT/DST algorithms;
- Channel migration;
- nonhomogeneous BCs and anchoring;
- second-order integration;
- control and differentiable stepping.
