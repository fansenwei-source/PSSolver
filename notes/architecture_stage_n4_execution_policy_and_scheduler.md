# Architecture Stage N.4: execution-policy and scheduler consolidation

## Scope

Stage N.4 consolidates the qualified Stage N.1--N.3 shadow architecture.  It
does not change the Beris--Edwards equations, the Plane Stokes solution,
boundary conditions, dealiasing, time integration, production defaults,
Channel, checkpoint formats, or benchmark `develop`.

The numerical candidate is intentionally the same Stage N.3 algorithm.  The
change is architectural: representation choice, field lifetime, transform
scheduling, and observability now have one owner each.

## One configuration authority

`AlgebraicExecutionPolicy` replaces the three internal feature booleans with
four valid, monotone modes:

1. `eager_componentwise`;
2. `generation_reuse`;
3. `lazy_componentwise`;
4. `batched_physical_islands`.

Impossible combinations cannot be represented by the policy.  The
experimental runtime builder still accepts the N.1--N.3 flags so archived
drivers and qualification code remain reproducible, but it translates them at
the builder boundary.  A policy and compatibility flags cannot be supplied in
the same call.  Downstream contexts and adapters receive only the resolved
policy.

## Responsibility boundaries

The consolidated path assigns the following ownership:

- `AlgebraicExecutionPolicy` owns the immutable representation strategy;
- `AlgebraicGenerationState` owns current-generation tensor lifetime and
  invalidation;
- dependency protocols owned by the model/solver declare physical consumers;
- `BoundarySignatureTransformScheduler` owns compatibility grouping and
  projected transform batching;
- `AlgebraicPhysicalIslandScheduler` owns cache-aware materialization and
  pointwise-output projection scheduling;
- legacy model/algebraic contexts remain numerical adapters for individual
  projected transforms and spectral differential operators;
- performance and materialization counters remain observational and cannot
  make a field current.

The execution context retains a small `forward_projected_many` compatibility
facade because the qualified Plane algebraic solvers already use that
contract.  The facade delegates batching to the scheduler; it does not contain
grouping logic.

## Complete transform compatibility key

Every scheduled batch records an immutable compatibility key containing:

- resolved boundary signature;
- forward or inverse direction;
- physical and spectral shapes;
- existing runtime batch size;
- real and spectral dtypes;
- device.

One scheduler is bound to one geometry/runtime context and one scheduling call
is bounded to one declared computation island.  Consequently no batch crosses
a geometry, generation, timestep, device, dtype, field-storage, or checkpoint
boundary.  Component order is reconstructed after grouped execution.

## Metadata and compatibility

Runtime metadata now contains the resolved execution-policy record and an
explicit scheduler record.  The historical representation-reuse,
materialization, and physical-island metadata remains present and derives from
the same policy, preserving existing N.1--N.3 analysis.

Production modules do not import or expose the new policy or scheduler.  The
new Stage N.4 trajectory, profiler, planner, and analyzer remain explicit
shadow entry points.

## Local qualification

The CPU qualification compares the Stage N.3 compatibility flags with the
consolidated N.4 policy path in one target worktree.  It requires exact
physical and spectral state equality, identical transform-call counts,
identical materialization lifecycle counters, zero retained representation
pairs, and preservation of the runtime batch dimension.

The bounded H100 qualification uses a detached Stage N.3 parent worktree as
the control and a detached Stage N.4 worktree as the candidate, so scheduler
extraction is measured across commits rather than by comparing two aliases of
the new implementation.  Both use the same frozen Stage M production input.
It requires:

- all 21 Q/u/p arrays at steps 0--6 to pass relative L2 `1e-10`, with pressure
  compared after demeaning;
- identical final profiler Q hashes;
- identical forward/inverse transform calls;
- identical materialization lifecycle counters;
- mean candidate/control timestep ratio at most `1.03`;
- at least two of three paired ratios at most `1.05`;
- allocated and reserved memory ratios at most `1.03`;
- no non-finite value, graph break, compile fallback, OOM, or CUDA error.

Passing authorizes design of a production-migration stage.  It does not itself
promote the shadow runtime or change a production default.
