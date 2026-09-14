# Architecture Stage N.2: lazy physical materialization

## Scope

Stage N.2 remains an opt-in experimental shadow path. It does not change the
production Plane driver, the generic solver defaults, Channel, checkpoint
formats, or the benchmark `develop` branch.

The Stage N.1 cache removed redundant physical-to-spectral transforms within
one algebraic DAG generation. It did not remove the inverse transform that was
performed immediately after every algebraic output. Stage N.2 makes those
physical values lazy while preserving the physical mapping contract seen by
existing algebraic solvers.

## Representation lifecycle

`AlgebraicGenerationState` owns the physical and spectral views for one
synchronized pre-explicit-RHS state:

- every evolved input starts with both physical and spectral representations;
- every algebraic output is first published in native spectral form;
- a legacy `state[name]` access materializes and memoizes the physical tensor;
- a representation-aware solver can request `spectral_dependency(...)`
  without triggering an inverse transform;
- all physical and spectral references are invalidated before the next
  algebraic generation;
- the state is not evolved, checkpointed, or reused across timesteps.

The exact-alias and tensor-version cache from Stage N.1 remains active during
the algebraic solve and still releases all tensor pairs at the end of the DAG.
The lazy transient mapping has the pre-existing lifetime of one synchronized
pre-RHS state so that the physical model can request only the transient fields
that its explicit RHS actually consumes.

## Plane-specific spectral consumers

The experimental Plane executors now avoid physical round trips when the next
operation is already spectral:

1. Q and velocity gradients consume the source spectrum directly.
2. The mixed-parity complete-stress divergence consumes the stored stress
   spectra directly, while preserving the qualified physical/spectral summation
   order and final force projection.
3. The Plane Stokes solver consumes the projected force spectra directly.

Pointwise nonlinear work is unchanged. Molecular field components and Q
gradients are still materialized for the stress kernel; velocity gradients are
still materialized for the explicit Beris--Edwards RHS. No physical dependency
is silently approximated or omitted.

## Local CPU evidence

On the complete small Plane Beris--Edwards DAG, over the same three audited
steps:

- Stage N.1 control: 73 forward and 121 inverse transform calls;
- Stage N.2 candidate: 67 forward and 73 inverse transform calls;
- physical materializations per current generation: 29;
- published components left unmaterialized: 25;
- maximum physical/spectral difference remains within the existing
  `1e-12` CPU equivalence gate.

The inverse-call reduction is the intended Stage N.2 mechanism. H100 timing is
not inferred from this CPU call audit and requires the bounded qualification.

## H100 qualification policy

The H100 control enables Stage N.1 representation reuse with eager algebraic
materialization. The candidate adds lazy physical materialization. Both use the
same frozen Stage M production reference and configuration.

The candidate must satisfy all of the following:

- 21 Q/u/p arrays for steps 0--6 and relative L2 at most `1e-10` against the
  frozen production trajectory, with pressure compared after demeaning;
- inverse transform calls strictly lower in every paired audit;
- forward transform calls no higher in every paired audit;
- at most 32 physical materializations and at least 20 deferred published
  components in every candidate audit;
- no Stage N.1 representation pair retained after a generation;
- mean H100 speedup at least `1.02x` and candidate faster in at least two of
  three balanced trials;
- peak allocated and reserved memory ratios no greater than `1.05`.

Passing yields `A_recommended` and permits Stage N.3 design only. It does not
authorize production promotion.
