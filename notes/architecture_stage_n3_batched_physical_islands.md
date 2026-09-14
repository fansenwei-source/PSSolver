# Architecture Stage N.3: batched physical computation islands

## Scope

Stage N.3 remains an opt-in experimental Plane shadow path. It does not
modify the production Plane driver, generic solver defaults, Channel,
checkpoint formats, benchmark `develop`, the Beris--Edwards equations,
boundary conditions, dealiasing, or time integration.

Stage N.2 avoided physical materialization for algebraic values whose next
consumer was spectral. The remaining physical consumers still requested one
component at a time, and pointwise island outputs were likewise projected one
component at a time. Stage N.3 groups those transforms by resolved boundary
signature while preserving every component's declared spectral space.

## Dependency contracts

Two optional execution contracts define the physical islands:

- `AlgebraicPhysicalDependenciesProtocol` lets a geometry-dispatched solver
  identify which of its declared algebraic dependencies it reads physically.
- `ExplicitRHSPhysicalDependenciesProtocol` lets a physical model identify
  which state components its explicit RHS reads.

These are execution hints, not new mathematical dependencies. A declared
physical dependency must already belong to the corresponding model or
algebraic-system specification. Spectral-only Plane gradient, force-divergence,
and Stokes consumers declare no physical inputs; molecular-field and stress
pointwise kernels declare the inputs they actually read.

## Boundary-compatible batching

Within one algebraic generation, a physical island is prefetched before its
pointwise kernel runs. Missing spectra are grouped only when they have exactly
the same resolved periodic/DCT/DST boundary signature. Each group is
concatenated along the existing batch dimension, transformed once, and split
back into exact component views. Pointwise outputs from molecular field,
stress, force, and the explicit Q RHS use the analogous grouped forward path.

No group crosses a boundary signature, generation, timestep, geometry, dtype,
device, checkpoint, or persistent field-storage boundary. The Stage N.1 exact
alias/version guard remains in force, and Stage N.2 invalidation still releases
all transient physical and spectral references before the next generation.

## Diagnostics

Stage N.3 extends the generation-local materialization snapshot with:

- physical materialization batches;
- components transformed in multi-component batches;
- singleton batch count;
- maximum transform batch size;
- physical-island prefetch count and requested-component count.
- on-demand materializations outside a declared physical island.

Component materialization count remains distinct from transform-batch count.
This makes it possible to verify that Stage N.3 changes scheduling rather than
silently dropping a physical dependency.

## Local CPU evidence

For the complete small Plane Beris--Edwards DAG over the same three audited
steps:

- Stage N.2 control: 67 forward and 73 inverse transform calls;
- Stage N.3 candidate: 13 forward and 23 inverse transform calls;
- candidate current generation: 29 physical components in 4 inverse batches;
- all 29 transformed components participate in multi-component batches;
- maximum inverse batch size: 15 components;
- physical and spectral state agree within the existing `1e-12` CPU gate.

These counts establish the intended mechanism only. H100 throughput and memory
must be measured by the bounded qualification.

## H100 qualification policy

The H100 control is the qualified Stage N.2 lazy-materialization runtime. The
candidate adds declared, boundary-signature-batched physical islands. Both use
the immutable Stage M production reference and identical numerical settings.

Recommendation requires all of the following:

- all 21 Q/u/p arrays at steps 0--6 pass relative L2 `1e-10` against the
  frozen production trajectory, with pressure compared after demeaning;
- forward and inverse transform calls both decrease in every paired audit;
- the number of physically materialized components does not increase;
- at least 20 components are batched and maximum batch size is at least 5;
- no candidate materialization occurs outside a declared physical island;
- no Stage N.1 representation pair survives a generation;
- mean H100 speedup is at least `1.02x` and at least two of three paired trials
  are faster;
- peak allocated and reserved memory ratios are at most `1.05`.

Passing yields `A_recommended` and permits Stage N.4 architecture
consolidation. It does not authorize production promotion.
