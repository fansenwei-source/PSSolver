# Phase 7 P7.7.2: boundary assignment and simulation composition

Status: `P7_7_2_COMPLETE_P7_7_3_NOT_AUTHORIZED`.

Parent baseline: `56c2682fbd1a237fc9c1deb69c8e2d7183da6a9b` on
`next/pssolver-v0.2.0-architecture`.

Classification:
`PASS_P7_7_2_BOUNDARY_ASSIGNMENT_SIMULATION_COMPOSITION`.

P7.7.2 adds the tensor-free composition seam promised by the Phase 0 charter.
It changes no equation, coefficient, physical boundary law, grid, timestep,
transform, solver, runtime, output, checkpoint, or production default. Neither
the Plane nor Channel application consumes this seam yet.

The machine-readable authority is
[phase_7_p772_simulation_composition.json](phase_7_p772_simulation_composition.json).

## Face-aware boundary declarations

`pssolver.core.boundary` now contains four provisional declaration objects:

- `BoundarySide` distinguishes lower and upper sides of an axis;
- `FaceBoundaryCondition` attaches one law to one oriented face slot;
- `ComponentBoundaryAssignment` groups every face law for one scalar field
  component;
- `BoundaryAssignment` owns the complete component-to-face map for one
  simulation.

The representation supports different laws at the lower and upper sides of a
bounded axis. This is the structural requirement needed later for strong
anchoring, asymmetric walls, or nonhomogeneous lifting, but P7.7.2 does not
introduce or qualify any of those new physical laws. The existing boundary
condition classes still admit homogeneous periodic, Dirichlet, and Neumann
contracts only.

Periodic topology is represented by paired periodic face slots so that every
component has one uniform axis/side schema. These are the identified sides of
a periodic seam, not physical walls.

`BoundarySemantic` prevents pressure-space compatibility from being confused
with prescribed wall physics:

- `physical` is used for Q and velocity;
- `algebraic_compatibility` is used for the pressure Lagrange multiplier.

The Plane `distortion_odd_z` entry is intentionally absent from physical
`BoundaryAssignment`. It is a derived modal parity required by the legacy
complete-stress lowering and is recorded under discretization compatibility
metadata instead. P7.7.2 therefore does not promote DCT/DST parity to a
physical boundary condition.

## Tensor-free `SimulationSpec`

The new canonical provisional module is
`pssolver.configuration.simulation`. It composes:

```text
EquationSystemSpec
+ GeometrySpec
+ BoundaryAssignment
+ NumericsConfig
+ TimeIntegrationSpec
+ discretization parameters
+ InitialConditionSpec
+ ExecutionSpec
+ WorkflowSpec
+ InvocationSpec
---------------------------
SimulationSpec
```

`SimulationSpec` performs construction-time semantic validation:

1. geometry and boundary dimensions must agree;
2. boundary assignments cover exactly evolved and algebraic components;
3. transient constitutive intermediates cannot receive direct boundaries;
4. evolved components require physical boundary semantics;
5. periodic axes require periodic conditions on both sides;
6. bounded axes reject periodic conditions;
7. generated initial-condition families must be declared by the equation
   system.

It remains a declaration, not an executable model. It does not contain or
construct tensors, transform plans, basis choices, solvers, workspaces,
runtime state, or callbacks.

## Time, execution, workflow, and initial-condition ownership

`TimeIntegrationSpec` wraps the existing tensor-free `IntegratorSpec` and a
refresh policy. Both qualified adapters currently select the existing
projected semi-implicit Euler identity. Plane additionally preserves the
resolved spectral-refresh declaration; Channel has no corresponding refresh
policy.

`ExecutionSpec` records the requested backend family, runtime path, device,
compile, cache, and implementation selectors. It does not instantiate that
runtime.

`WorkflowSpec` records duration, output/observation schedules, checkpoint and
restart inputs, and output locations. `InvocationSpec` records non-scientific
invocation provenance.

`InitialConditionSpec` distinguishes generated realizations from external
snapshots. Generated families must be declared by the equation model;
snapshot input is represented honestly as `external_snapshot`, not relabeled
as a generated model family.

## Four independent identities

Each `SimulationSpec` emits deterministic SHA-256 records for:

- scientific identity: equations, effective physical parameters, physical
  dimensions/topology, and field/face boundary semantics;
- discretization identity: grid, grid placement, precision, dealiasing,
  storage/transform intent, time integration, refresh, and algebraic solver
  controls;
- execution identity: backend, runtime path, device and implementation
  policy;
- run identity: initial-condition realization, duration, schedules, restart,
  output, and invocation provenance.

Tests prove the separation directly: changing only workflow duration changes
only run identity; changing only a device selector changes only execution
identity; refining shape at fixed physical lengths changes only
discretization identity. Adapter compatibility metadata changes the complete
declaration hash but none of the four authoritative identities.

## Plane compatibility composition

`compose_plane_beris_edwards_simulation` maps the qualified component graph to:

- complete-stress Beris--Edwards equations;
- `PlaneSlab` topology;
- Q Neumann anchoring on the bounded axis;
- free-slip tangential velocity Neumann conditions;
- no-penetration normal velocity Dirichlet conditions;
- algebraic pressure Neumann compatibility;
- the existing numerical and projected semi-implicit Euler policies;
- the existing extruded-defect-gas realization;
- the existing execution, workflow, and invocation values.

All five Q components, three velocity components, and pressure are assigned.
No gradient, molecular-field, stress, or force intermediate receives a direct
physical boundary.

## Channel compatibility composition

`compose_channel_active_nematics_simulation` maps the qualified component
graph to:

- legacy active-force active-nematic equations;
- `RectangularChannel` topology;
- Q Neumann anchoring on both bounded axes;
- no-slip Dirichlet velocity conditions on both bounded axes;
- algebraic pressure Neumann compatibility;
- the existing numerical and projected semi-implicit Euler policies;
- the pressure-PCG tolerance, iteration, and warm-start contract;
- generated or snapshot initial-condition provenance;
- the existing execution and workflow values.

The two adapters retain the complete source component metadata as an explicit
non-authoritative compatibility record. This makes the composition auditable
and lossless while preventing legacy mixed-ownership metadata from redefining
the four identities.

## Dependency and API status

The only new configuration-to-system edge is:

```text
pssolver.configuration.simulation
  -> pssolver.systems.equations
```

It is recorded exactly in the import-boundary ratchet. The new declaration and
adapter modules import no Torch, NumPy, backend, transform, operator, linear
solver, runtime, workflow implementation, or application module.

P7.7.2 adds no package-root export. Direct leaf-module imports remain a
provisional architecture surface. Existing public facades, pickle identity,
metadata schema, application construction, and runtime defaults are
unchanged.

## Verification

The P7.7.0--P7.7.2, core-contract, and import-boundary target set passed:

```text
59 passed
```

The complete local CPU suite passed:

```text
2079 passed, 8 subtests passed
```

No test was skipped or deselected. No GPU qualification is required because
the new declarations are not imported by a runtime and cannot enter a
timestep.

## Performance effect

There is no hot-path cost. Existing applications do not import or construct
`SimulationSpec`, and no runtime consumes it. Even after a future connection,
composition and capability resolution must happen once before device binding;
the timestep must receive pre-bound callables and fixed layouts rather than
perform field, boundary, or registry dispatch.

## Next slice

P7.7.3 is **not authorized by this result**. When separately authorized, it
should implement capability resolution and tensor-free lowering from the
composed declaration to exact basis, derivative, nullspace, and
geometry-specific solver requirements. Unsupported model/geometry/boundary
combinations must fail before tensor allocation.

P7.7.3 must not connect a new production runtime, alter defaults, or claim
that complete-stress Beris--Edwards has already been qualified in Channel.
New boundary physics and Phase 8 remain out of scope.
