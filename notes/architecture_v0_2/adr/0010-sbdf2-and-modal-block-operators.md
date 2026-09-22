# ADR 0010: SBDF2 and bounded modal block operators

Status: accepted for Phase 4.

## Context

Phase 3 established explicit runtime state, bounded workspace ownership, a
pre-bound step program, exact restart behavior, and qualified Plane runtime
connections.  The only production time integrator remains projected
semi-implicit Euler, and its linear implicit update is represented by a
component-wise spectral denominator.  That is sufficient for the current
Plane application but does not demonstrate that the architecture can host a
second integration scheme or a genuinely coupled linear evolution operator.

Adding a general symbolic equation system, arbitrary variable coefficients,
or Newton--Krylov machinery at this point would mix several independent
architectural questions.  Phase 4 instead needs the smallest second consumer
that exercises integrator history and coupled per-mode linear algebra.

## Decision

The second time integrator is constant-step SBDF2 for a frozen split

```text
du/dt = L u + N(u).
```

For every step after startup it satisfies

```text
(3 u[n+1] - 4 u[n] + u[n-1]) / (2 dt)
    = L u[n+1] + 2 N(u[n]) - N(u[n-1]).
```

The first successful step uses the already qualified projected
semi-implicit Euler program.  A first-order startup has local error of order
`dt**2` and is sufficient for a globally second-order SBDF2 trajectory.  This
startup choice, the transition to multistep mode, and the history update order
are observable metadata and restart contracts.

Phase 4 supports constant `dt` only.  Changing `dt` after history exists is
rejected rather than silently rescaling or discarding history.  CNAB2 remains
a later integrator and is not an alias for SBDF2.

The SBDF2 persistent history consists of the previous evolved native spectrum
and the previous explicit native-spectral right-hand side.  It belongs to
runtime state, not generation-local workspace.  Candidate current RHS values,
linear-solve RHS values, and solve scratch remain bounded workspace.  History
is committed only after projection, inverse reconstruction, scheduled refresh,
and progress commit all succeed; a failed step leaves state and history at the
last successful generation.

The first block-linear capability is a fixed two-component, mode-independent-
in-topology but mode-dependent-in-coefficients spectral system:

```text
A(k) x(k) = b(k),    A(k) in C^(2 x 2).
```

Planning owns the tensor-free coupling and component-order declaration.
Device binding owns coefficient tensors, validation, selected solve
implementation, and any cached factorization.  Runtime workspaces own solve
scratch.  The first qualification covers block size two only; it does not
claim a general block algebra API.  Unsupported block sizes and singular or
non-finite systems fail before a timestep.

The qualifying model is a two-component constant-coefficient periodic
reaction--diffusion system with an analytic Fourier-mode solution.  It tests
both off-diagonal coupling and diffusion without introducing a new geometry,
physical boundary mechanism, or algebraic Stokes solve.

## Compatibility

- projected semi-implicit Euler remains the production default;
- `legacy_production` remains the Plane runtime default;
- the Plane checkpoint-v1 schema and its byte layout remain unchanged;
- SBDF2 initially uses a separate provisional generic checkpoint contract;
- restart across different integrator identities is rejected;
- no Phase 4 type is promoted at the package root before closure;
- Phase 4 does not connect SBDF2 or the block operator to Plane.

## Qualification

The phase must demonstrate first-order Euler and second-order SBDF2 temporal
convergence against an analytic solution, same-backend continuous/restart
equivalence, exact history restoration, fail-closed corrupt-history and
cross-integrator gates, finite CPU/GPU execution, bounded workspace behavior,
and one balanced H100 non-regression qualification of the optimized SBDF2 and
block-solve implementations.

## Deferred work

This decision does not authorize CNAB2, adaptive or variable timesteps,
arbitrary block size, spatially variable implicit coefficients, symbolic DSLs,
Newton/Krylov solvers, complete Plane migration, strong anchoring, Channel,
control, or any production-default promotion.

