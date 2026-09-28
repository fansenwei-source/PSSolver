# Phase 9 P9.2: batch-one periodic activity-control functional runtime

Status: `PASS_P9_2_PERIODIC_ACTIVITY_FUNCTIONAL_RUNTIME`.

Baseline: `dc7177fb95bc94797aa995553d986a9229b14378` on
`next/pssolver-v0.2.0-architecture`.

P9.2 implements the first executable member of the provisional
`pssolver.functional` API.  It is intentionally limited to the already
qualified complete-stress Beris--Edwards model in a fully periodic box, one
batch, eager pointwise kernels, disabled spectral refresh, and a spatial
activity control.  Unsupported combinations fail during construction.  The
stable package-root API and every production default remain unchanged.

The machine-readable authority is
[phase_9_p92_periodic_functional_runtime.json](phase_9_p92_periodic_functional_runtime.json).

## Explicit state and discrete map

The functional state is the flat tuple

1. `q_physical`, with layout `(5, batch, nx, ny, nz)`; and
2. `q_spectral`, with the backend's native full-complex or Hermitian-packed
   spectral layout.

The spectrum is not a disposable cache.  Production periodic execution with
spectral refresh disabled advances and retains it independently, so omitting
it would lose a persistent value that can affect the next step through
roundoff.  Carrying both representations makes the functional state complete
and permits exact reconstruction of the qualified discrete map.

Each call binds fresh internal field storage around the supplied tensors and
uses the production runtime's complete-stress model, periodic direct Stokes
solver, tensor-product transforms, dealiasing projector, IMEX timestep,
denominator, and coefficients.  The caller's tensors are never updated in
place.  Q-gradient reuse and pressure diagnostics are disabled in the
functional runtime so no trajectory-dependent cache or diagnostic state can
affect a later call.

## Activity-control semantics

The only P9.2 control is `activity`, with exact shape
`(batch, nx, ny, nz)`, exact runtime dtype and device, cell-centered placement,
and a model-owned lower bound of zero.  Invalid shape, dtype, device, NaN,
Inf, or negative values are rejected; the runtime never converts or clips a
control.  Each call passes activity through a local control mapping and does
not mutate the production model's parameter registry.

The active contribution remains the existing complete-stress expression

`div(beta * alpha * Q)`.

The spatial `beta * alpha * Q` product is formed before the spectral
divergence.  The dealias rule and projected-transform execution mode are part
of the control and runtime identity.  A nonuniform activity field therefore
includes its spatial derivative through the divergence of the product; it is
not approximated as `alpha * div(beta * Q)`.

## Observations and capabilities

Named observations are:

- `Q`, available with or without a terminal control;
- `velocity`, computed by the direct periodic Stokes solve under the current
  activity; and
- `pressure`, with the zero-mean pressure gauge under the current activity.

`step_and_observe(state, controls, step_index)` returns the next state and the
observation of the input state under the current control.  It evaluates the
flow once.  Its observation is bitwise identical to a separate `observe`
call.  `observe(state, None)` returns only `Q`, because velocity and pressure
are control-dependent algebraic fields.

The runtime declares pure-step semantics, the combined operation, batch size
one, PyTorch autograd connectivity for state and activity, and differentiation
through the direct Fourier Stokes solve.  It does not claim qualified bitwise
replay, a durable checkpoint bridge, explicit JVP/VJP, or a custom adjoint.
Those remain later Phase 9 slices.

## Construction and ownership

`periodic_activity_functional_request(simulation)` produces the exact control
and observation declarations for a canonical `SimulationSpec`.
`build_functional_runtime(request)` validates the existing public periodic
compiler contract, loads the same snapshot bytes as the production
application, binds the same qualified runtime, and rejects any noncanonical
request.  Functional construction performs no output-directory writes.

The reusable snapshot loader is now named
`load_periodic_initial_q`; production and functional construction share it.
The legacy `pssolver.control` package is unchanged, PSSolver imports no
external control project, and optimizer/objective ownership remains outside
PSSolver.

## Verification and non-claims

CPU tests establish exact state layout, JSON identity, fail-closed capability
claims, input immutability, terminal-observation behavior, spatial-control
sensitivity, autograd reachability, strict input rejection, and exact
single-step Q/velocity/pressure agreement with the qualified production
periodic runtime.  Full-suite and source-hygiene results are recorded in the
machine-readable record.

P9.2 does not qualify deterministic multi-step replay, checkpoint conversion,
finite-difference gradients, hidden-detach coverage, production/functional
tolerance envelopes, H100 performance, larger batches, a Channel functional
runtime, an optimizer, or an independent consumer.  In the frozen Phase 9
order, P9.3 is next: deterministic replay and the functional/durable state
bridge.
