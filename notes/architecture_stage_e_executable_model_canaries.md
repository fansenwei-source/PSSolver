# Architecture Stage E: executable scalar-model canaries

## Scope

Stage E introduces the smallest execution-facing physical-model contract that
can run real timesteps through the frozen `SpectralPlan` and the opt-in Stage D
legacy assembly.  It deliberately does not migrate active nematics, Plane or
Channel Stokes solvers, production benchmark drivers, numerical defaults,
checkpoint formats, or the qualified transform/integrator implementations.

The two canaries are:

- scalar diffusion,
  `partial_t phi = diffusivity * Laplacian(phi)`;
- Allen--Cahn reaction diffusion,
  `partial_t phi = diffusivity * Laplacian(phi)
  + linear_reaction * phi - cubic_reaction * phi^3`.

They cover a zero explicit term and a nonlinear physical-space term without
introducing coupled algebraic fields or geometry-specific pressure semantics.

## Dependency direction

The new execution boundary is:

```text
canary model
    -> ModelExecutionContext (coordinates and mathematical Laplacian)
    -> ProblemSpec
    -> SpectralPlan
    -> opt-in pssolver.experimental adapter
    -> existing SpectralSolver / projector / integrator
```

The model does not receive `SpectralSolver`, legacy `Fields`, transform names,
spectral masks, or a projector.  It declares fields and physical parameters,
constructs initial physical values and diagonal linear operators, and evaluates
an explicit physical-space right-hand side.  The adapter alone converts that
right-hand side to the runtime's native spectral representation and applies the
projector selected by the numerical plan.

`ModelExecutionContext` exposes only:

- physical and spectral shapes;
- domain lengths and cell-centred one-dimensional axis coordinates;
- real dtype, device, and batch size;
- Laplacian eigenvalues addressed by declared component name.

The one-dimensional coordinate representation avoids materializing a full mesh
for every axis.  Model implementations must treat coordinate, Laplacian and
state tensors supplied through the execution contract as read-only.

## Runtime adapter

`build_experimental_model_runtime` is explicitly imported from
`pssolver.experimental`; it is not exported from the production `pssolver`
package.  It performs the following ordered construction:

1. validate `ExecutableModelProtocol`;
2. construct `ProblemSpec` and immutable `SpectralPlan`;
3. materialize the Stage D legacy assembly;
4. create and shadow-check the existing transform backend and projector;
5. create the restricted execution context;
6. request model initial values and linear spectral operators;
7. declare legacy runtime fields;
8. wrap `explicit_rhs` in a legacy `torch.nn.Module`;
9. build the unchanged current solver and re-run the plan/runtime shadow check.

Explicit RHS output must be an exact component-name mapping.  Each tensor must
match its state tensor's shape, dtype, and device.  No transform or boundary
choice can be returned by a physical model.

Stage E rejects algebraic fields instead of guessing update order or coupling
semantics.  That restriction is intentional: active-nematic velocity,
pressure, Schur solves, and geometry-specific zero-mode policies require a
separate, explicit execution contract.

## Canary models and initial conditions

`ScalarDiffusionModel` and `AllenCahnModel` live in
`pssolver.models.canary`.  Their initial condition is a tensor-product basis
eigenmode constructed from physical boundary semantics:

- periodic axis: cosine with wavenumber `2*pi*m/L`;
- homogeneous Neumann axis: cosine with wavenumber `pi*m/L`;
- homogeneous Dirichlet axis: sine with wavenumber `pi*m/L`.

This keeps initial-condition convention and physical parameters in the model,
while grid coordinates and boundary-to-basis resolution remain outside it.

The existing non-benchmark `examples/diffusion.py` now demonstrates this
experimental path.  It has a `main()` guard and imports the opt-in builder
explicitly.  No benchmark driver imports the experimental package.

## Qualification evidence

Focused Stage E tests compare each assembled canary with an independently
constructed legacy solver:

- periodic scalar diffusion with full-complex storage and no dealiasing;
- mixed periodic/Neumann Allen--Cahn on a PlaneSlab with Hermitian storage,
  cubic-half dealiasing, and truncated projected transforms.

Initial physical fields, initial spectra, diagonal linear operators, and eight
semi-implicit timesteps must be bitwise identical.  Further tests verify the
mathematical RHS, plan metadata, context isolation, failure on algebraic fields,
failure on invalid RHS mappings, and absence of the new builder from the
production package API.

Desktop CPU qualification after the final implementation:

- Stage A--E focused architecture suite: `83 passed`;
- complete repository suite: `699 passed, 8 subtests passed`.

Both runs used `PYTHONDONTWRITEBYTECODE=1` and disabled the pytest cache.  No
GPU qualification is required because Stage E neither changes nor replaces a
qualified production execution path.

## Rollback and next boundary

Stage E changes no solver, transform, integrator, Plane/Channel runtime, or
active-nematic implementation.  Its runtime effects are reachable only through
an explicit experimental import.  Rolling it back removes the new execution
contracts, canary package, model adapter, tests and this note, and restores the
single migrated example.

The next stage should first generalize lifecycle and algebraic-field semantics
behind another opt-in boundary.  Active-nematic migration should begin only
after update ordering, geometry-owned Stokes dispatch, state/diagnostic views,
and rollback-compatible trajectory tests are specified.  It must not infer
those semantics from the legacy `Fields` container.
