# Architecture Stage J: coupled Beris--Edwards Plane evolution

## Scope and result

Stage J turns the static-Q Stage I qualification graph into a complete,
opt-in Beris--Edwards/Stokes timestep for the Plane geometry.  The frozen
pre-RHS dependency order is now

```text
Q -> H -------------------\
 \-> grad(Q) -> stress ----> force -> Plane Stokes -> u,p -> grad(u)
  \______________________________________________________________/
                              |
                              v
                    complete explicit Q RHS
```

Only Q is evolved.  Velocity and pressure are instantaneous stored algebraic
fields.  H, Q gradients, both stress parts, force, and velocity gradients are
transient values for one synchronized pre-RHS state.  No production entry
point, benchmark driver, numerical default, Plane solver, or Channel solver
selects this path.

## J1: complete Q equation and IMEX split

`BerisEdwardsPlaneCoupledModel` reuses the already qualified compact tensor
convention and the production equation helpers.  It evolves

```text
(partial_t + u dot grad) Q - S(W, D, Q) = H / gamma,
```

with the complete flow-alignment and co-rotation term `S`.  The implementation
retains the production semi-implicit split:

```text
implicit:  (-A Q + L1 Laplacian(Q)) / gamma
explicit:  B/C bulk relaxation, material advection,
           flow alignment, and co-rotation.
```

The diagonal linear symbol is produced by the same
`beris_edwards_linear_operator()` helper used by the benchmark driver.  The
explicit physical components are produced by
`beris_edwards_q_nonlinear_components()`.  Stage J therefore adds no second
Q convention and no independently rederived evolution law.

`rotational_viscosity` is explicit model data.  The raw molecular field in the
constitutive stress remains undivided by this value; only the Q relaxation
equation uses `H/gamma`.

## J2: velocity-gradient dependency after Stokes

The Q right-hand side needs all nine components `partial_i u_j`.  Stage J
declares them as a separate transient algebraic system with the capability

```text
beris_edwards_velocity_gradient
```

and registers an exact Plane implementation.  The generic tensor-gradient
executor is shared with the Q-gradient capability, but dispatch, component
ownership, dependency order, and boundary validation remain explicit.

The DAG planner places velocity gradients after the geometry-specific Stokes
solve even though the model declaration does not manually encode an execution
loop.  Each wall-normal derivative switches Neumann and Dirichlet parity as
required; periodic derivatives retain periodic parity.  A mismatched declared
output space is rejected before time integration.

The full coupled runtime contains:

- five evolved Q components;
- three stored algebraic velocity components and one stored pressure;
- fifty transient components: five H, fifteen Q gradients, eighteen stress,
  three force, and nine velocity gradients.

Only the nine evolved/stored components enter the legacy field container.
Transient values are not checkpointed.

## J3: dealiased evolved-state lifecycle

The production Plane benchmark projects the evolved Q spectrum after every
semi-implicit update and before the inverse transform.  Its initial and reset
states are also projected before algebraic fields are evaluated.  The previous
experimental bridge projected nonlinear transforms but used the base legacy
integrator for the evolved spectrum, so it was not yet a faithful coupled
trajectory path.

Stage J adds the opt-in `ProjectedSemiImplicitEulerIntegrator`.  When a
dealiasing projector is enabled, its order is:

1. synchronize the algebraic pre-RHS state;
2. evaluate the explicit nonlinear spectrum;
3. apply the semi-implicit Euler numerator and denominator;
4. project the updated evolved spectrum;
5. inverse-transform the projected state;
6. advance the existing spectral-refresh clock.

A scheduled spectral refresh rebuilds the evolved spectrum through the same
projected transform.  Initial construction and reset also project Q before
refreshing the constitutive/Stokes DAG.  With dealiasing disabled, the adapter
continues to use the ordinary legacy semi-implicit integrator.

Runtime metadata records the integration scheme, whether dynamic spectral
projection is active, and the concrete integrator class.  This makes the
state-lifecycle choice auditable instead of implicit.

Stage E and Stage F independent manual-reference tests were updated to use the
same qualified projected-state lifecycle.  Their physical models and expected
equations were not changed.

## J4: equation and trajectory qualification

Stage J tests establish:

1. the exact six-system execution order
   `H -> grad(Q) -> stress -> force -> flow -> grad(u)`;
2. the expected evolved, stored-algebraic, and transient component counts;
3. the production A/L1 diagonal IMEX symbol for every Q component;
4. all nine velocity gradients against the component-aware mathematical
   operator context;
5. the complete explicit Q RHS against the qualified equation helper;
6. initial, reset, and per-step spectral projection of evolved Q;
7. single-step and six-step trajectories against independent assembly of the
   existing production `BerisEdwardsQNonlinearModel` and
   `BerisEdwardsFreeSlipStokes` adapters;
8. both full-complex storage and the qualified Hermitian-half/truncated path;
9. finite, JSON-compatible lifecycle and IMEX metadata;
10. continued isolation from production drivers and public top-level defaults.

The cross-architecture trajectory comparison uses tight floating-point
tolerances rather than requiring byte identity between distinct operation
orderings.  The reference path uses the same physical coefficients, initial
projected Q state, Stokes zero-mode policy, transform storage, and integrator
semantics.

Desktop CPU qualification after the final implementation:

- architecture Stages B--J: `123 passed`;
- Stage F--J plus existing Beris--Edwards force, energy, stress, projected
  transform, static-transform, pressure, and free-slip Stokes suites:
  `171 passed`;
- complete repository suite: `758 passed, 8 subtests passed`.

All commands used an external Python bytecode cache and disabled the pytest
cache.  No GPU qualification is required because Stage J remains an opt-in
reference architecture and changes no production execution path or default.

## Production boundary after Stage J

Stage J demonstrates that the separated architecture can execute the complete
current Plane Beris--Edwards/Stokes timestep and reproduce the already
qualified production adapters for short CPU trajectories.  It does not yet
replace `Plane_beris_edwards_stokes.py`, claim checkpoint compatibility for a
migrated driver, or promote the reference executor as a performance path.

The next stage should introduce a shadow driver/runtime boundary around this
model: production-compatible initial-condition injection, save/checkpoint and
restart coordination, observation synchronization, and metadata comparison.
That path should first run side by side with the existing driver.  Only after
longer CPU trajectory and H100 performance/equivalence gates should production
selection be considered.

Channel constitutive parity, Channel-specific active-nematic boundary choices,
and any native runtime that removes the legacy bridge remain separate later
stages.  They must use their own geometry-specific registrations rather than
assuming that the qualified Plane operators apply unchanged.
