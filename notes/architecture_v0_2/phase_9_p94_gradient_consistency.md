# Phase 9 P9.4: gradient and production-consistency validation

Status: `PASS_P9_4_CPU_GRADIENT_AND_R12_VALIDATION`.

Baseline: `3bc3841` on `next/pssolver-v0.2.0-architecture`.

P9.4 closes the CPU qualification slice for gradients, hidden-detach
detection, checkpoint-stride gradient invariance, and requirement R12 for the
batch-one periodic complete-stress activity runtime.  It does not change a
production default, implement a control objective or checkpoint scheduler, or
claim a CUDA result.  The machine-readable authority is
[phase_9_p94_gradient_consistency.json](phase_9_p94_gradient_consistency.json).

## Frozen gradient audit

`pssolver.functional.validation` provides a fail-closed qualification API.
It uses fixed internal audit functionals, directions, finite-difference steps,
and tolerances; callers cannot supply an approximate override.  Float64 uses
central steps `1e-6` for state and `1e-4` for activity with relative tolerance
`2e-5`.  The activity audit requires enough interior distance from its
nonnegative admissibility boundary for both central samples.

The validator compares PyTorch reverse-mode directional derivatives with
central finite differences separately for the complete explicit state and
the spatial activity field.  It also constructs separate dynamics and
observation graphs and requires finite nonzero gradients for:

- dynamics from physical Q;
- dynamics from native spectral Q;
- dynamics from activity;
- velocity/pressure observations from physical Q;
- velocity/pressure observations from native spectral Q; and
- velocity/pressure observations from activity.

The separation is intentional: direct Q observation dependence cannot hide a
detach in the timestep map.  Negative proxies that detach either the state or
the control are rejected by the corresponding dynamics gate even though the
remaining graph is differentiable.

## Checkpoint-stride gradient invariance

An independent consumer-style reference performs a three-step forward pass,
saves detached explicit-state checkpoints, and replays each segment during a
reverse vector-Jacobian sweep.  Strides one, two, and three produce
bitwise-identical terminal state, initial-state cotangent, and every activity
gradient on the fixed CPU float64 execution identity.

This is a qualification oracle rather than a scheduler in PSSolver.  The
independent control project continues to own checkpoint placement, control
history, objectives, and optimization.  P9.3 already proved that its durable
serialization boundary preserves the same explicit state bytes.

## R12 production/functional consistency

The R12 validator consumes one production adapter at the same explicit input
state and applies the same spatially varying activity tensor.  It compares
input physical and spectral Q, input-state Q/velocity/pressure observations,
and next physical and spectral Q.  The frozen float64 envelope is
`atol=5e-13`, `rtol=5e-12`; neither tolerance is a function argument.

On the qualified CPU case every compared tensor was byte-for-byte identical.
The formal contract nevertheless remains the strict predeclared tolerance
envelope rather than an unconditional cross-device byte-identity promise.
A deliberate `1e-6` production-state perturbation returns a failed report,
and the validating entry point raises `FunctionalValidationError` carrying
that report.

## Scope and non-claims

The validation report objects are JSON-serializable through `to_metadata()`
and remain provisional under `pssolver.functional`; the stable package-root
API is unchanged.  P9.4 does not add an optimizer, objective, scheduling
policy, explicit JVP/VJP API, custom adjoint, larger batch, Channel pressure
adjoint, or external control-project dependency.

CPU tests cover the positive gradient audit, state/control hidden-detach
negatives, bitwise checkpoint-stride replay gradients, positive R12
consistency, deliberate R12 mismatch, and the absence of tolerance override
parameters.  The complete Phase 9 focused set passed 51 tests.  The complete
CPU suite passed 2478 tests plus 8 subtests with no failures, skips,
deselections, or expected failures; `git diff --check` passed.  CUDA/H100
gradients, VJP cost, peak memory, and larger batches remain P9.5.
