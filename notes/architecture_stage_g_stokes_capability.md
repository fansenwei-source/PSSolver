# Architecture Stage G: geometry-specific incompressible Stokes capability

## Scope and result

Stage G adds the first non-scalar algebraic capability to the opt-in execution
architecture.  It defines a backend-independent incompressible Stokes request,
then dispatches that request to the already qualified Plane and Channel modal
solvers through two separate registrations.

This stage does **not** migrate the Beris--Edwards model or a production
benchmark driver.  `Plane_beris_edwards_stokes.py` remains the benchmark
reference.  The new path is qualified with a small body-force canary whose
velocity and pressure results are compared directly with the fixed legacy
geometry solvers.

## G1: one physical request, four algebraic outputs

`IncompressibleStokesSystemSpec` declares

```text
dependencies:       force_x, force_y, force_z
algebraic outputs:  ux, uy, uz, p
capability:         incompressible_stokes
```

and the physical/run choices

```text
viscosity
friction
pressure gauge
tangential zero-mode policy
```

Component order is coordinate order.  Every force, velocity and pressure
component keeps its own `BoundarySet` in the model's field declarations.  The
Stokes request neither selects FFT/DCT/DST transforms nor names a concrete
solver.

The typed request lowers to the generic `AlgebraicSystemSpec` used by Stage F
and is parsed and revalidated at the solver-factory boundary.  Unknown or
missing parameters, invalid component layouts, non-finite coefficients and
inconsistent policy/friction combinations fail before a solve.

## G2: pressure gauge is not a tangential zero-mode policy

Stage G represents the two choices with different enums:

- `PressureGauge.ZERO_MEAN` fixes the additive pressure constant;
- `TangentialZeroModePolicy.ZERO_MEAN` removes the Plane uniform tangential
  velocity/force mode when friction is zero;
- `TangentialZeroModePolicy.FRICTION` retains and determines that mode through
  positive Brinkman friction;
- `TangentialZeroModePolicy.NOT_APPLICABLE` states that the geometry has no
  such null mode.

The Plane registration accepts only `zero_mean` with zero friction or
`friction` with positive friction.  This preserves the important modeling
fact that deleting the mean tangential mode is not a pressure gauge choice.

The current rectangular Channel uses homogeneous Dirichlet velocity on both
bounded directions.  Its velocity operator has no uniform tangential null
mode, so the Channel registration requires `not_applicable`.  A non-negative
friction coefficient may still be part of its physical Brinkman operator, but
it is not used there as a null-mode gauge.

Both current implementations support only the zero-mean pressure gauge.
Unsupported policies fail explicitly; no silent replacement is made.

## G3: component-wise spaces and exact geometry dispatch

The current Plane implementation is registered as

```text
(PlaneSlab, incompressible_stokes)
    -> plane_free_slip_modal_stokes
```

and requires

```text
force_x, force_y, ux, uy : periodic / periodic / Neumann
force_z, uz              : periodic / periodic / Dirichlet
p                        : periodic / periodic / Neumann
```

The current Channel implementation is registered independently as

```text
(RectangularChannel, incompressible_stokes)
    -> channel_no_slip_modal_stokes
```

and requires

```text
force_x, force_y, force_z, ux, uy, uz
                         : periodic / Dirichlet / Dirichlet
p                        : periodic / Neumann / Neumann
```

Force and corresponding velocity components must use the same native basis.
Every boundary signature is checked during construction.  `PeriodicBox` has no
Stokes registration, and Plane/Channel policy or boundary crossovers are
rejected.  Exact dispatch from Stage F remains in force; there is no geometry
fallback.

The Plane adapter reuses `FreeSlipModalStokesSolver.solve_force_hats()`.  The
Channel implementation now exposes the equivalent public
`ModalSaddleStokesCompute.solve_force_hats()` method.  Its production
`forward()` delegates to this method while preserving the previous arithmetic
operation order.  Public pressure-gradient and divergence methods make
equation-level validation possible without treating private implementation
details as the future contract.

## G4: numerical controls remain outside the physical model

Pressure residual collection for Plane and iterative pressure controls for
Channel are runtime implementation settings:

- `PlaneStokesSolverOptions.pressure_diagnostics`;
- `ChannelStokesSolverOptions.pressure_relative_tolerance`;
- `ChannelStokesSolverOptions.pressure_max_iterations`;
- `ChannelStokesSolverOptions.pressure_fixed_iterations`.

They configure the geometry registrations and are not embedded in the
physical Stokes request.  This separates model parameters from solver choices
and lets future backends expose different algorithms without changing the
model contract.  The resolved values are nevertheless retained in solver
observability metadata, so a run remains auditable.

## G5: diagnostics are observations, not PDE state

The optional `InspectableAlgebraicSolverProtocol` provides a diagnostic
snapshot separately from the model field layout.  The Stokes snapshot records
the pressure iteration count, residual, relative residual, whether a warm
start was used and the number of solves.

Disabling Plane pressure residual measurements records JSON `null` values,
not NaN.  Runtime metadata describes whether diagnostics are enabled and which
pressure algorithm is resolved.  Diagnostic mappings are checked for finite,
JSON-compatible values before being returned.

No residual, iteration counter or profiling value is added to `FieldSpec`,
`Fields.spatial` or `Fields.spectral`.  The canary storage remains exactly the
three evolved forces plus four physical algebraic outputs.

## G6: warm-start state and restart provenance

The direct Plane Schur solve is explicitly stateless.  The Channel pressure
solve is iterative and stores `pressure_guess` as an implementation warm
start.  Stage G exposes capture and restore operations that:

1. keep warm-start tensors separate from physical state fields;
2. clone captured tensors;
3. record system, capability and exact implementation identity;
4. record tensor shape, dtype, capture device and SHA-256;
5. require an exact system/dispatch match during restore;
6. validate shape, dtype, device, finiteness and pressure gauge;
7. clear implementation warm starts during an experimental runtime reset.

`AlgebraicRuntimeRestartState` is the solver-state portion of a future complete
checkpoint.  It does not claim to replace evolved-field, time-integrator or
random-number checkpoint state.  A future checkpoint coordinator must save
those layers together.

## G7: qualification canary

`BodyForceStokesCanaryModel` evolves three body-force components by scalar
diffusion and requests instantaneous velocity and pressure.  This gives every
timestep a changing Stokes input without importing active-nematic physics.

The Stage G characterization suite verifies:

- Plane and Channel results against separately constructed, fixed legacy
  geometry solvers;
- bitwise equality of all four native spectral outputs and physical fields at
  initial synchronization, stale post-step state and explicit observation
  synchronization over four timesteps;
- zero pressure gauge, incompressibility and momentum residuals;
- distinct Plane zero-mean and friction behavior for constant tangential force;
- diagnostics remaining outside state storage;
- disabled-diagnostic JSON behavior;
- Channel warm-start capture, SHA provenance, restore and identity rejection;
- explicit Plane stateless restart behavior;
- exact Plane/Channel registrations and rejection of crossovers;
- typed, immutable Stokes request validation;
- preservation of the Channel pre-Stage-G arithmetic operation order;
- continued isolation of the benchmark driver and root production API.

Desktop CPU qualification after the final implementation:

- Stage A--G architecture plus existing Plane/Channel Stokes suites:
  `142 passed`;
- complete repository suite: `725 passed, 8 subtests passed`.

Both runs used a separate Python bytecode cache and disabled the pytest cache.
No GPU qualification is required for Stage G because it does not select a new
production path or change a numerical default.  The sole lower-level
production-file edit is the behavior-preserving extraction of Channel force
spectra solving into `solve_force_hats()`, protected by an exact operation-order
comparison.

## Production boundary after Stage G

Stage G qualifies the Stokes execution seam, but it is not yet a
Beris--Edwards migration.  In particular it does not yet define:

- a model-level construction of the complete nematic force;
- reusable tensor-gradient and stress-divergence operator capabilities in the
  new architecture;
- active-nematic parameter and initialization migration;
- complete simulation checkpoint coordination;
- production metadata compatibility and long-trajectory comparison;
- a production replacement for either benchmark driver.

Those items belong to the next migration stage.  Until they pass their own
equation-level, trajectory and restart gates, the existing benchmark code
remains authoritative.
