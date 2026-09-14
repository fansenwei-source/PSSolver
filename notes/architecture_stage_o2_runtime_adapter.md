# Architecture Stage O.2: opt-in Plane runtime adapter

## Scope and outcome

Stage O.2 introduces the first executable dual-path boundary for the Plane
Beris--Edwards model.  It does not promote the separated architecture and does
not modify Channel, the generic `SpectralSolver` construction API, scientific
equations, boundary conditions, defaults, or benchmark `develop`.

`PlaneBerisEdwardsRunSpec` is still the sole configuration authority.  It now
owns one `PlaneRuntimePath` value:

- `legacy_production`, the omitted-selector default and rollback oracle;
- `separated_canary`, available only through an explicit request.

Requested and effective values are written additively to metadata.  There is
no fallback and no lower layer may reinterpret the selection.

## Runtime boundary

`pssolver.runtime.plane_beris_edwards` defines a narrow protocol covering only
the surface needed before Stage O.3 supplies a shared workflow:

- the underlying solver and projector;
- field access;
- timestep advancement;
- algebraic synchronization for observation;
- additive runtime metadata.

The legacy adapter wraps the production assembly without changing its
operators or timestep implementation.  Its builder is invoked only after the
request's configuration identity and runtime-selection metadata agree exactly
with the immutable run specification.

The separated adapter imports `pssolver.experimental` lazily and constructs the
already-qualified Stage N.4.1 policy (`AlgebraicExecutionPolicy.batched()`).
An omitted/default selection therefore does not import experimental modules.
The canary uses the same raw initial tensors and performs its own projected
reset before numerical execution.

## Deliberate O.2 limits

Diagnostics and the legacy Q-gradient-cache compatibility flag are rejected
for the separated canary before solver construction.  Their common ownership
belongs to the Stage O.3 workflow/observation migration.  This is an explicit
failure, not a silent fallback.

Output-loop and completion behavior remain in the production script in O.2.
Checkpoint/restart compatibility is not claimed.  Stage O.3 must move those
responsibilities behind a common, versioned boundary before O.4 qualification.

## Characterization

The focused tests establish that:

- omitted selection remains `legacy_production`;
- programmatic and CLI selection share one authority;
- mixed metadata/spec authorities fail before a builder is called;
- the legacy path does not import the experimental architecture;
- Channel and generic solver entry points do not acquire Plane dispatch;
- an explicit one-step CPU canary produces the same projected initial Q and
  agrees with legacy Q/u/p to floating-point roundoff.

The local legacy characterization compared this change directly with the O.1
commit `f63f2f022c7a0ddddf660f694970c0ac514ec6cf`: every Q/u/p file at steps
0, 1, and 2 was byte-identical.  A separate 12x10x8, float64, six-step
legacy/canary comparison used the same projected-initial-Q SHA-256 and had a
maximum raw relative-L2 difference of `3.78683332213882875e-16` across all
saved Q/u/p arrays.  These are local characterization results, not the Stage
O.4 H100 production-canary qualification.

No default promotion is authorized.  The next architectural action is Stage
O.3: unify workflow, observation, completion, and same-backend restart
boundaries while preserving the legacy output contract.
