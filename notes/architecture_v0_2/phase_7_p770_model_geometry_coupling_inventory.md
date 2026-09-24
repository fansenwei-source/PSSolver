# Phase 7 P7.7.0: model--geometry coupling inventory

Status: `P7_7_0_COMPLETE_P7_7_1_NOT_AUTHORIZED`.

Baseline: `b8ce70842b6da0fb45aa1558d5e183dac9117990` on
`next/pssolver-v0.2.0-architecture`.

This is a declaration-only and audit-only slice. It changes no equation,
boundary condition, transform, timestep, runtime, output, checkpoint schema,
or production default. It does not authorize P7.7.1 or Phase 8.

The machine-readable authority is
[phase_7_p770_model_geometry_coupling_inventory.json](phase_7_p770_model_geometry_coupling_inventory.json).

## Finding

PSSolver currently has two well-isolated and extensively qualified vertical
application slices:

```text
PlaneBerisEdwardsRunSpec
  -> Plane component graph
  -> Plane Beris--Edwards application
  -> Plane runtime adapter
  -> free-slip Plane Stokes lowering
  -> Plane workflow/checkpoint/output

ChannelActiveNematicRunSpec
  -> Channel component graph
  -> Channel active-nematic application
  -> Channel runtime adapter
  -> no-slip Channel Stokes/Schur lowering
  -> Channel workflow/checkpoint/output
```

That is a substantial improvement over one monolithic script: configuration,
runtime selection, mutable state, workflow, and geometry-specific solvers now
have visible seams. It is not yet the orthogonal composition promised by the
architecture charter:

```text
EquationSystemSpec + GeometrySpec + BoundaryAssignment + NumericsSpec
                                      |
                                SimulationSpec
                                      |
                           capability resolution
                                      |
                     tensor-free DiscretizationPlan
                                      |
                       device-bound ExecutionPlan
```

The current application names are therefore accurate descriptions of
qualified presets, not interchangeable model and geometry axes.

## A crucial scientific distinction

The existing Plane and Channel applications do not contain the same physical
model on two geometries.

| Application | Q/force model | Geometry and flow solve | Qualified status |
|---|---|---|---|
| Plane | five-component Beris--Edwards Q dynamics with the complete one-constant reactive, active, and distortion nematic stress | one bounded wall-normal axis; free-slip modal Stokes; Plane tangential-zero-mode policy | qualified |
| Channel | legacy rho-parameterized active-nematic Q dynamics with active-force-only Stokes forcing | one periodic and two bounded axes; no-slip Channel Stokes/Schur; pressure PCG warm start | qualified |

Consequently, replacing `PlaneSlab` with `RectangularChannel` is not a valid
way to obtain “Beris--Edwards in a Channel.” That new combination needs an
honest complete-stress equation declaration, wall-axis constitutive
derivatives, boundary lowering, a compatible Channel Stokes force path, and
its own manufactured and trajectory qualification. Conversely, the Channel
legacy active-force model has not been declared or qualified on Plane.

This distinction must survive the migration. P7.7 is intended to enable new
combinations, not to erase real scientific differences between existing
ones.

## Existing shared foundations

The repository already contains useful horizontal layers:

- `pssolver.core` owns domain, geometry, boundary, field, and numerical value
  objects;
- `pssolver.geometries` represents Plane and Channel topology without choosing
  a physical boundary condition or solver;
- `pssolver.models.active_nematics` contains the Q convention, reusable
  pointwise Beris--Edwards kernels, field names, and initial-condition helpers;
- `pssolver.systems.stokes` contains a tensor-free incompressible Stokes
  request, pressure gauge, and tangential-zero-mode policy;
- transform, backend, execution-state, workspace, and integrator contracts are
  already shared below the application-specific construction paths;
- the two geometry-specific Stokes solvers are correctly separate and should
  remain first-class implementations.

The missing layer is not another numerical kernel. It is a composition and
lowering layer that decides whether a model, geometry, boundary assignment,
and numerical policy are compatible, and selects an exact implementation
without putting registry lookup in the timestep.

## Plane ownership inventory

The supported flat facade contains 54 fields. The current component graph has
11 members. P7.7.0 records all of them so that later migration cannot silently
drop or duplicate an authority.

| Intended owner | Flat facade fields |
|---|---|
| model and preset | `activity_number`, `parameterization`, `frank_k`, `coefficient_min`, `coefficient_max`, `ldg_a`, `ldg_b`, `ldg_c`, `gamma`, `flow_alignment`, `beta` |
| geometry and grid | `height`, `lx`, `ly`, `nx`, `ny`, `nz` |
| Stokes system | `eta`, `zero_mode_policy`, `friction_mode_fric` |
| time discretization | `dt`, `spectral_refresh` |
| numerics | `dtype`, `dealias_rule`, `projected_transform_execution`, `transform_execution_order`, `spectral_storage` |
| initial condition | `seed`, `num_defect_pairs`, `defect_min_separation`, `defect_core_radius`, `background_angle`, `twist_amplitude`, `twist_modes`, `initial_s` |
| execution | `device`, `molecular_field_linear_space`, `stress_divergence_sum_space`, `pointwise_execution`, `tf32`, `disable_q_gradient_reuse`, `runtime_path` |
| workflow | `output_dir`, `steps`, `save_start_step`, `save_interval`, `diagnostic_interval`, `diagnostics`, `save_hydrodynamics`, `checkpoint_interval`, `restart_from` |
| invocation/provenance | `validation_config_sha256`, `dry_run` |
| boundary assignment | `boundaries` |

The component graph currently contains:

```text
geometry
boundaries
effective_boundaries
numerics
physics
preset
time_stepping
initial_condition
execution
workflow
invocation
```

The most important Plane couplings are:

1. The composition root requires exactly a three-dimensional `PlaneSlab`
   with periodic axes `(0,1)` and bounded axis `2`.
2. `PlaneFreeSlipBoundaryConditions` combines physical Q/velocity conditions
   with derived pressure and distortion-stress modal parity. It is not yet a
   pure field-to-face physical boundary assignment.
3. Shendruk coefficient resolution jointly consumes Plane wall height,
   material coefficients, and benchmark parameterization.
4. `PlaneBerisEdwardsPhysicsSpec` vertically bundles material, preset request,
   Stokes request, and friction-mode bookkeeping.
5. The extruded-defect-gas initial condition validates twist modes directly
   against Plane `nz`.
6. Hermitian-half storage is tied to the last periodic Plane axis.
7. The application directly composes complete nematic stress with the
   free-slip modal Stokes implementation.
8. `zero_mean` versus `friction` resolves the uniform tangential free-slip
   mode. It is a physical/reference-frame choice distinct from pressure gauge.
9. Runtime paths and runtime adapters are Plane-specific.
10. Observation, checkpoint, restart identity, and completion ordering are
    Plane-specific workflow contracts.

## Channel ownership inventory

The supported flat facade contains 40 dataclass fields, including the private
derived component cache. The current component graph has 10 members.

| Current component owner | Flat facade fields |
|---|---|
| geometry and grid | `shape`, `lengths` |
| time discretization | `dt` |
| model | `rho`, `elastic_constant`, `activity`, `beta`, `flow_alignment` |
| Stokes system | `friction`, `viscosity` |
| pressure solver | `pressure_relative_tolerance`, `pressure_max_iterations`, `pressure_fixed_iterations` |
| numerics | `dealias_rule`, `projected_transform_execution`, `transform_execution_order`, `spectral_storage` |
| initial condition | `seed`, `initial_s`, `noise_theta`, `noise_phi`, `smoothing_sigma`, `initialization_mode`, `snapshot_mode`, `snapshot_directory`, `snapshot_step` |
| execution | `batch_size`, `device`, `dtype`, `runtime_path` |
| workflow | `steps`, `save_interval`, `diagnostics_enabled`, `diagnostic_interval`, `checkpoint_interval`, `restart_from`, `generated_output_directory`, `snapshot_output_directory` |
| boundary assignment | `boundaries` |
| derived component cache | `_components` |

The component graph currently contains:

```text
geometry
boundaries
numerics
material
stokes
pressure_solver
initial_condition
execution
workflow
dt
```

The most important Channel couplings are:

1. The composition root requires a `RectangularChannel` whose only periodic
   axis is streamwise axis `0`.
2. `ChannelActiveNematicMaterialSpec` derives Landau--de Gennes coefficients
   from `rho` and vertically packages Channel model parameters.
3. `ChannelBoundaryConditions` stores physical laws as legacy parity strings
   rather than field-component laws attached to geometric faces.
4. The builder freezes no dealiasing, full projected transforms, real-first
   execution, and full-complex storage for the Channel oracle.
5. The application enforces float32, `flow_alignment=1`, and `batch_size=1`.
6. The package application hard-codes `aligned_x_smooth_noise`; the facade's
   legacy snapshot mode is not a supported package initial-state path.
7. The application directly composes active-force-only forcing with the
   no-slip Channel Stokes implementation.
8. Pressure PCG controls and its persistent warm-start pressure guess are
   Channel-specific numerical and restart requirements.
9. Runtime paths, adapters, and the compiled application bridge are
   Channel-specific.
10. Observation, checkpoint, pressure-state restoration, diagnostics, and
    completion ordering are Channel-specific workflow contracts.

## Boundary and nullspace inventory

The two applications intentionally have different physical and algebraic
contracts.

| Quantity | Plane | Channel |
|---|---|---|
| topology | periodic, periodic, bounded | periodic, bounded, bounded |
| Q | periodic/periodic/Neumann | periodic/Neumann/Neumann |
| tangential velocity | periodic/periodic/Neumann | periodic/Dirichlet/Dirichlet |
| normal velocity | periodic/periodic/Dirichlet | periodic/Dirichlet/Dirichlet |
| pressure | periodic/periodic/Neumann modal multiplier | periodic/Neumann/Neumann, zero-mean gauge |
| uniform tangential velocity mode | unresolved without `zero_mean` or positive friction | not a null mode under no-slip walls |
| pressure algorithm | direct Plane modal Schur complement | Channel PCG Schur solve with warm start |

DCT/DST or parity names are not promoted to physical boundary conditions.
They remain possible lowering choices after a physical field/face assignment
has been validated against geometry and solver capabilities.

## Cross-application gaps

The audit found seven structural gaps:

1. There is no complete geometry-neutral `EquationSystemSpec` for production
   Q evolution, constitutive force, algebraic Stokes output, fields, and
   diagnostics.
2. There is no common `SimulationSpec` composing model, geometry, boundary,
   initial condition, numerics, integration, execution, workflow, and
   invocation identities.
3. There is no capability resolver that selects a lowering from exact model,
   geometry, boundary, basis, nullspace, and solver requirements.
4. Plane and Channel boundary declarations use different representations and
   mix physical and modal information differently.
5. Applications still assemble metadata, choose initial conditions, select
   model/solver combinations, and construct runtimes.
6. Runtime and checkpoint adapter identities are application-specific. This is
   safe today but prevents a shared composition root.
7. Unsupported combinations lack one structured pre-allocation explanation;
   rejection is distributed among application-specific validators.

## Target ownership after P7.7

The following boundaries are now frozen for later slices:

- `EquationSystemSpec` owns fields, physical parameters, evolution and
  constitutive laws, algebraic subsystem requests, required capabilities,
  model diagnostics, and model-owned initial-condition families.
- `GeometrySpec` owns topology, axes/faces, coordinates, metrics, wall
  normals/tangents, grid placement, and separability capabilities.
- `BoundaryAssignment` attaches physical laws and values to field components
  and geometric faces. Strong anchoring belongs here, not in a geometry name.
- `NumericsSpec` owns precision, dealiasing, storage, transform policy, and
  discretization intent.
- `TimeIntegrationSpec` owns `dt`, integrator family, history, and refresh.
- `ExecutionSpec` owns backend, device, compile/cache policy, concrete
  implementation selectors, and runtime path.
- `WorkflowSpec` owns duration, schedules, observation/output,
  checkpoint/restart, and completion ordering.
- `SimulationSpec` composes these tensor-free declarations, validates
  cross-component meaning, and separates scientific, discretization,
  execution, and run identity.
- capability lowering selects basis/parity, derivative maps, lifting/tau,
  and an exact geometry-specific Poisson, Helmholtz, Stokes, or Schur
  implementation. Unsupported combinations fail before tensor allocation.
- existing application classes become compatibility preset adapters and CLI
  translation roots. They must not retain independent PDE or solver logic.

## Migration constraints

1. Existing Plane and Channel applications remain supported compatibility
   presets and rollback oracles throughout P7.7.
2. The two present physical models remain explicitly distinct.
3. Geometry-specific fast solvers remain first-class; abstraction must not
   force them through a slower universal solver.
4. Physical boundary laws remain separate from transform implementation.
5. Capability resolution happens at construction, never in the hot loop.
6. Unsupported combinations fail explicitly and never silently fall back.
7. Scientific, discretization, execution, and run identities remain separate;
   restart remains fail-closed.
8. Legacy runtime defaults remain unchanged.
9. P7.7 adds no new physical boundary and does not start Phase 8.

## Next slice: P7.7.1

P7.7.1 is **not authorized by this inventory**. When separately authorized,
its scope should be limited to geometry-neutral equation-system declarations:

1. introduce tensor-free field, model, constitutive-force, and algebraic
   system declarations;
2. represent the two existing physical models honestly as distinct specs;
3. adapt existing Plane and Channel component graphs to those declarations;
4. preserve public facades, metadata, construction order, runtime paths,
   numerical output, and defaults.

Its exit gate is declaration parity, not a new runtime. Every field and
parameter must have one owner; declarations must not import Torch, transforms,
runtime, workflows, applications, or geometry-specific solvers; and existing
Plane and Channel metadata must be reconstructible without loss.
