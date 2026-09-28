# Phase 9 P9.6: first independent periodic functional consumer

## Purpose

P9.6 qualifies the P9.5 batch-one periodic activity runtime through one real,
independently maintained control consumer.  It tests whether the provisional
public API is sufficient outside PSSolver without moving optimization code,
objectives, checkpoint scheduling, or adjoint orchestration into PSSolver.
The machine-readable authority is
[phase_9_p96_independent_consumer_plan.json](phase_9_p96_independent_consumer_plan.json).

The dependency remains strictly one way: the consumer may depend on an
installed PSSolver wheel; PSSolver does not import, name, vendor, or inspect
the consumer repository at runtime.

## Adapter boundary

The consumer must implement its adapter using only `pssolver.functional` and
the ordinary public `pssolver` construction objects.  It must not access
private attributes such as `_solver`, `_adapter`, `_request`, or any legacy
`pssolver.control` object.

The adapter performs these explicit translations:

- `FunctionalStateSpec.components` becomes the consumer's ordered tuple of
  tensor specifications, preserving shape, dtype, allocated device, and order;
- the single public `activity` field in `control_specs` becomes the consumer's
  tensor control input;
- calls to `step` and `step_and_observe` wrap that tensor as
  `{"activity": value}`;
- the public observation mapping is forwarded without renaming `Q`,
  `velocity`, or `pressure`;
- `FunctionalRuntimeIdentity.to_metadata()` becomes the consumer identity
  payload without dropping scientific, discretization, execution, or state
  layout blocks;
- `FunctionalCapabilitySet` is validated semantically, including pure step,
  combined observation, `torch_autograd`, batch size one, and bitwise replay;
  capability names are not guessed from object internals.

The adapter fails closed on unsupported API version, more than one control
field, any batch size other than one, non-bitwise replay, fallback, shape,
dtype, device, identity, or admissible-bound mismatch.  It never converts
tensors implicitly and has no legacy fallback.

## Real consumer qualification

The consumer constructs the frozen R128 periodic complete-stress application
through public PSSolver declarations and uses a spatially varying activity
field.  Its real checkpointed discrete-adjoint loop must execute a quadratic
`Q`-tracking objective for a short, fixed horizon.  The test is an interface
and numerical integration qualification, not a control-science result and not
an optimization campaign.

CPU preflight uses a small periodic grid.  It verifies the adapter contract,
one-step combined observation semantics, no input mutation, a finite nonzero
state and activity gradient, directional finite differences, bitwise forward
replay, and bitwise gradient invariance for checkpoint strides 1, 2, and 4.
It also rejects a tampered identity and a runtime whose deterministic replay
is not bitwise.

The formal H100 qualification uses one Slurm job, one R128 problem, float64,
TF32 off, batch size one, and no automatic retry.  It builds and imports clean
installed wheels for both provider and consumer, rejects source shadowing, and
pins both Git commits and both wheel hashes.  It repeats the real forward,
checkpoint replay, adjoint, stride-invariance, finite-difference, finite,
memory, and provenance gates.  No optimizer iteration is required.

## Pass condition and nonclaims

P9.6 passes only when the independent consumer can use PSSolver's public
functional API with no private access or fallback, and its checkpointed
adjoint agrees with the frozen validation contracts.  A passing consumer
authorizes P9.7 planning.  It does not stabilize the provisional API, qualify
Channel pressure differentiation, authorize batch sizes above one, promote a
runtime, change a production default, or establish a scientific control
result.
