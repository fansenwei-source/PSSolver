# Phase 9 P9.7.0: Channel functional-runtime contract freeze

## Outcome

P9.7.0 freezes the next implementation sequence without changing runtime
code.  Its machine-readable authority is
[phase_9_p970_channel_functional_plan.json](phase_9_p970_channel_functional_plan.json).
The first P9.7 combination is the existing complete-stress Beris--Edwards
model in the rectangular Channel: periodic in `x`, bounded in `y` and `z`,
homogeneous Neumann `Q`, no-slip velocity, Neumann pressure, and a zero-mean
pressure gauge.  It remains float64, TF32-off, and batch one.

P9.7 does not build a second Channel solver.  It places a pure functional
contract around the qualified production ingredients already present in
`channel_beris_edwards.py` and `channel_no_slip.py`, then validates the
pressure derivative before exposing the path to a second independent
consumer.

The JSON inventory records the P9.7.0 byte identities of those ingredients.
The runtime, pressure solver, and provisional functional contract are planned
change surfaces, so their recorded hashes are a baseline rather than an
immutability rule.  The legacy functional step, DAL, and Channel DAL test are
frozen oracles and remain byte-bound unless a separately qualified
compatibility fix is authorized.

## State and hidden-state decision

The first functional Channel state contains only physical and native-spectral
`Q`.  The pressure warm-start guess is not state: it is disabled for the
functional path, and every pressure solve starts from the canonical zero
guess.  This avoids a history-dependent algebraic cache, gives the step a
single-valued mathematical meaning, and prevents an approximate PCG stopping
path from silently becoming part of the differentiable state.  Existing
production warm-start behavior is not changed.

The `Q` gradient cache is derived and rebuilt per call.  Diagnostics are
observations only and cannot affect a future step.  Progress counters remain
external.  Inputs may not be mutated.  `step_and_observe` must be equivalent
to a separate pure step and observation call, and same-identity replay must be
bitwise on the same device and dtype.

This is deliberately the smallest honest contract.  A later separately
qualified extension may introduce an explicit algebraic-solver state, but it
must then declare component-level differentiability and prove that replay and
gradients have the intended semantics.

## Pressure differentiation

The existing Channel pressure solve is a matrix-free Schur-complement PCG
solve with a zero-mean gauge.  P9.7 must not backpropagate through every PCG
iteration and must not infer a transpose from apparent symmetry.  The
production design is a custom implicit adjoint:

1. solve the primal problem `S p = b` in the zero-mean pressure subspace;
2. expose an explicit transpose action and verify its dot-product identity;
3. solve `S^T lambda = p_bar` in the same gauge during reverse mode;
4. propagate the resulting cotangents through the pressure right-hand side
   and the surrounding velocity/stress operations;
5. reject primal or adjoint results that do not satisfy their residual
   contracts.

A small-grid, fixed-iteration, unrolled autograd solve is retained only as an
oracle for the custom VJP.  The qualified production reverse pass must not
retain the PCG iteration graph, so its memory use must be independent of the
number of primal PCG iterations.  Linearity, zero-mean invariance,
transpose-dot-product, manufactured primal and transpose solves, implicit-VJP
agreement, finite differences, Taylor convergence, and memory are all hard
gates.

## Controls and observations

The first control is a non-negative, spatially varying activity field.  The
solver continues to form the physical product proportional to
`activity * Q` before taking the active-stress divergence; no alternate
discretization is introduced.  `Q`, velocity, and pressure are public
observations.  A terminal observation without a supplied control exposes only
`Q`, because velocity and pressure are control-dependent algebraic outputs.

Static `Q` Dirichlet lifting and Robin boundaries are not part of this first
Channel consumer because the frozen legacy Channel DAL oracle uses
homogeneous Neumann `Q`.  They are not removed or disallowed; they require a
later combination-specific qualification if a real consumer needs them.

## Ownership and the second consumer

PSSolver owns the Channel state, control, observation, identity, pressure
operator and transpose, implicit adjoint, admissible physical control, and
production/functional consistency.  The independent Control project owns its
adapter, control parameterization, objective, checkpoint schedule, adjoint
orchestration, optimizer, and campaigns.  The dependency remains one-way:
the consumer imports an installed PSSolver wheel; PSSolver never imports the
consumer.

The legacy `pssolver.control` Channel DAL and its tests remain frozen numerical
oracles.  They receive no new features.  The second consumer migrates the
exercise to the public `pssolver.functional` API and may not use private
members, source-checkout shadowing, `pssolver.control` as a runtime dependency,
or a fallback path.  Passing the interface qualification is not a scientific
control result.

## Implementation sequence

P9.7 is divided into independently reviewable slices:

1. P9.7.1 adds Channel functional declarations, request/state/control/
   observation specifications, and identity metadata without claiming a
   pressure gradient.
2. P9.7.2 adds the pressure transpose protocol and its linearity, gauge,
   dot-product, and manufactured tests.
3. P9.7.3 adds the custom implicit pressure adjoint and compares it with the
   small-grid unrolled oracle.
4. P9.7.4 connects the pure Channel step, activity control, observations,
   replay, and checkpoint bridge.
5. P9.7.5 qualifies the second consumer in the independent Control project.
6. P9.7.6 performs clean installed-wheel CPU and one-job H100 closure,
   including production/functional consistency, gradients, replay/restart,
   negative gates, memory, and provenance.

## Authorization boundary

P9.7.0 is complete and P9.7.1 is eligible for a separate implementation
authorization.  P9.7.0 itself does not authorize or implement P9.7.1--P9.7.6,
run HPCC work, change defaults, stabilize the provisional API, add larger
batches, or begin a control campaign.  Phase 9 remains incomplete.
