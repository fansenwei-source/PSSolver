# Phase 9 P9.7.3: Channel implicit pressure adjoint

## Result

P9.7.3 adds a custom first-order VJP for the batch-one Channel pressure solve
and validates it against a small-grid, fixed-iteration, unrolled CPU oracle.
The machine-readable authority is
[phase_9_p973_channel_pressure_implicit_adjoint.json](phase_9_p973_channel_pressure_implicit_adjoint.json).

This remains an additive, provisional facility in `pssolver.functional`.
The frozen production Channel solver, pressure PCG, warm-start path, Channel
runtime, defaults, and public simulation entry points are unchanged.  The
independent PSSolver-Control repository is also unchanged.

## Implicit differentiation contract

For fixed operator identity and a pressure right-hand side `b`, the forward
problem is

```text
S p = b.
```

The custom backward receives a pressure cotangent `p_bar` and solves

```text
S* b_bar = p_bar
```

with the explicit transpose action qualified in P9.7.2.  Both solves start
from zero.  Neither solve reads or writes the production pressure warm start.
Only `b` is differentiable: geometry, basis matrices, viscosity, friction,
preconditioner, tolerance, and iteration policy are frozen runtime identity.

The custom autograd node saves no PCG iteration tensors.  Forward and backward
execute their iterative solves without constructing an iteration graph, so
the retained autograd graph is independent of the number of PCG iterations.
The rule is intentionally first-order only; a higher-order derivative request
fails instead of silently differentiating a different algorithm.

## Independent oracle

`unrolled_channel_pressure_solve_oracle()` differentiates the fixed sequence
of PCG operations directly.  It is deliberately restricted to CPU tensors
with at most 512 complex modes and requires an explicit positive iteration
count.  It is qualification code, not a production fallback or runtime path.

On the frozen `(3, 2, 2)` pressure problem, five fixed PCG iterations recover
the small system to roundoff.  The custom implicit forward matches the
unrolled forward byte-for-byte, and its VJP agrees with the unrolled autograd
VJP within the registered float64 tolerance.

## Qualification

The P9.7.3 tests establish:

- manufactured pressure recovery with the custom forward solve;
- explicit-transpose VJP identity;
- implicit VJP agreement with the fixed-iteration unrolled CPU oracle;
- central finite-difference agreement and second-order gradient-subtracted
  Taylor remainders;
- no mutation of the input, production pressure guess, or frozen solver;
- zero tensors saved by the custom node and iteration-independent graph size;
- explicit rejection of higher-order differentiation;
- fail-closed shape, dtype, finiteness, iteration-count, and oracle-size
  contracts.

P9.7.3 still does not expose an executable Channel functional runtime or an
activity gradient.  It only qualifies differentiation of the pressure
right-hand side for fixed operator identity.  No H100 run is needed because
no production hot path or full Channel timestep is connected.

P9.7.4 is the next separately authorized planning boundary: connect the pure
Channel Q step, spatial activity control, aligned Q/velocity/pressure
observations, replay, and checkpoint bridge while preserving the zero-warm-
start pressure policy.  P9.7.3 does not authorize that implementation,
P9.7.5--P9.7.6, an H100 job, default promotion, or Phase 9 closure.
