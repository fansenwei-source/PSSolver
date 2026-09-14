# Architecture Stage N.1: generation-local representation reuse

## Scope

Stage N.1 is the first bounded optimization of the retained experimental Plane
shadow architecture.  It does not modify the production Plane driver, Channel,
the generic spectral solver, the Beris--Edwards equations, boundary conditions,
dealiasing, time integration, or any production default.

The Stage N H100 diagnostic found that pre-RHS algebraic synchronization used
about 92.27% of the instrumented timestep.  The same physical algebraic value
was often inverse-transformed for a downstream pointwise kernel and then
forward-transformed again when a later spectral solver required it.  Stage N.1
therefore addresses representation lifetime before attempting buffer pools or
operator-level micro-optimizations.

## Candidate design

`AlgebraicRepresentationCache` stores exact physical/spectral alias pairs only
while one frozen algebraic DAG generation is active.  A generation is seeded
from the synchronized evolved fields and their already-current spectra.
Materialized algebraic outputs register their physical value together with the
native spectrum returned by their owning solver.

A projected forward or inverse operation may reuse a value only if:

- the component name matches;
- the tensor is the exact same Python tensor object;
- both physical and spectral in-place mutation counters are unchanged; and
- the same algebraic generation is still active.

All tensor references are released when the generation completes or aborts.
No representation pair survives a timestep boundary, enters persistent field
storage, or appears in a checkpoint.  Only integer hit/miss counters remain
available as diagnostics.

The candidate is selected with
`enable_algebraic_representation_reuse=True`.  Its default is `False`, so the
qualified Stage N shadow path remains unchanged and serves as the H100 control.

## Local qualification

CPU tests cover exact alias and mutation guards, exceptional cleanup, metadata,
reset behavior, transform-call reduction, and control/candidate numerical
agreement.  In the small complete Plane Beris--Edwards DAG, three steps reduced
forward transforms from 137 to 73 while inverse transforms remained at 121.
The maximum observed control/candidate differences were approximately
`2.8e-17` in physical storage and `5.9e-16` in spectral storage.

## Bounded H100 gate

One later H100 job may run:

1. one six-step candidate trajectory from the immutable Stage M production
   reference and compare all 21 Q/u/p arrays at relative L2 `1e-10`;
2. three balanced control/candidate performance trials with ten warm-up and
   twenty timed steps; and
3. a separate two-step transform-call audit for every profile.

The candidate is recommended only when the trajectory passes, every candidate
trial reduces forward-transform calls without increasing inverse calls, no
generation retains tensor pairs, mean speedup is at least 1.02 with at least
two thirds of paired trials faster, and allocated/reserved memory ratios are at
most 1.05.  Passing this gate authorizes only the next optimization design; it
does not promote the shadow runtime into production.
