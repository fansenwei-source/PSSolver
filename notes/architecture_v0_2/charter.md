# PSSolver v0.2 architecture charter

Status: accepted for the `next/pssolver-v0.2.0-architecture` migration branch.

Code and numerical baseline: annotated tag `v0.1.2`, commit
`4fa614616e9d67c98be52c150eaf303a0c80b5c1`.

This charter governs architecture work after v0.1.2.  It does not change the
supported Plane equations, numerical defaults, runtime selector, output
contract, or production hot path.

## Mission

PSSolver is a GPU-first, spectral-first framework for smooth continuum PDEs on
regular or separable tensor-product geometries.  Its intended strength is the
combination of explicit scientific semantics, auditable spectral
discretization, geometry-specific fast solvers, reproducible workflows, and a
low-allocation GPU execution path.

The framework is intended to support multiple continuum models, physical
boundary laws, time integrators, and separable geometries without pretending
that one algebraic solver or one performance policy is optimal for all of
them.

## Supported architectural domain

The target domain includes:

- periodic boxes, slabs/planes, and rectangular channels;
- Fourier, cosine, sine, and mixed tensor-product spectral bases;
- diagonal spectral operators and small per-mode block systems;
- geometry-specific Poisson, Helmholtz, incompressibility, and Stokes solves;
- active-nematic, diffusion, reaction--diffusion, and phase-field models;
- CPU reference execution and optimized single-GPU execution;
- versioned metadata, restart, checkpoint, and validation contracts;
- a later functional execution path for differentiation and control.

The near-term architecture does not claim arbitrary unstructured geometry,
general finite elements, shocks, AMR, distributed decomposition, a symbolic
PDE language, or a stable third-party plugin ABI.

## Governing design principle

The architecture has a clear control plane and a flat data plane:

```text
ModelSpec + GeometrySpec + BoundaryAssignment + NumericsSpec
                              |
                          ProblemSpec
                              |
                        compile_problem
                              |
           tensor-free DiscretizationPlan / SpectralPlan
                              |
                   bind backend and device
                              |
          device-bound ExecutionPlan + WorkspacePlan
                              |
              RuntimeState + frozen StepProgram
                              |
                   Integrator and Workflow
```

High-level objects describe and validate a problem before execution.  Device
binding lowers that description into fixed layouts, direct callables, cached
resources, and workspaces.  The timestep loop must not rediscover dependencies
or trade contiguous batched operations for object-level generality.

## Normative object boundaries

### ProblemSpec

`ProblemSpec` expresses scientific meaning: model, parameters, geometry,
physical boundary assignments, grid, and discretization intent.  It is
tensor-free and device-independent.

### DiscretizationPlan

The tensor-free plan freezes bases, mode ordering, retained modes, derivative
maps, dealiasing, zero-mode policy, logical field layout, algebraic
dependencies, and required solver capabilities.  The current `SpectralPlan`
is the v0.1.2 precursor; Phase 0 does not rename it.

### ExecutionPlan

The execution plan is bound to a backend, device, dtype, and implementation
policy.  It owns immutable wavenumbers, transform resources, masks, selected
solver implementations, pre-bound callables, and workspace allocation plans.

### RuntimeState

Runtime state is the sole owner of mutable evolved state, time, integrator
history, representation validity, and explicitly declared persistent
algebraic state.  Reconstructible gradients, stresses, forces, and other
generation-local values belong to workspaces.

### Workflow

Workflows own stepping schedules, observations, output, restart, checkpoint,
completion ordering, and provenance.  They do not implement PDE equations or
spectral transforms.

## Model, geometry, and boundary separation

- Models declare fields, parameters, evolution laws, constitutive laws,
  algebraic systems, diagnostics, and model-owned initial conditions.
- Geometries declare topology, coordinates, metrics, wall normals/tangents,
  grid placement, and separability.
- Boundary assignments attach physical laws to field components and geometric
  faces.
- Planning lowers the combined meaning to FFT/DCT/DST, lifting, tau, or other
  admissible numerical mechanisms.

A Plane geometry does not imply free slip.  A Q-tensor model does not imply a
Neumann boundary.  DCT/DST names are implementation choices, not physical
boundary conditions.  Geometry-specific Stokes solvers are intentional parts
of the architecture rather than failures of abstraction.

## Module direction

The target dependency direction is:

```text
core
  -> geometries / model declarations
  -> planning
  -> backends / operators / linear solvers
  -> execution / integrators / runtime
  -> workflows / I/O
  -> applications
```

The executable subset of this rule is in
`tests/test_architecture_import_boundaries.py`.  Current v0.1.2 debts are
listed as eight exact edges.  Broad layer exemptions are forbidden.

| Exact v0.1.2 debt | Planned removal |
|---|---|
| `models.active_nematics.stokes -> transforms` | model force/solver composition split after Phase 1 |
| `configuration.plane_beris_edwards -> plane` | configuration-identity split |
| `configuration.plane_beris_edwards -> transforms` | move numerical defaults into specifications after Phase 1 |
| `runtime.plane_legacy -> integrator` | compiled-runtime adapter phase |
| `runtime.plane_legacy -> plane` | compiled-runtime adapter phase |
| `runtime.plane_legacy -> solver` | compiled-runtime adapter phase |
| `runtime.plane_legacy -> transforms` | Phase 1 canonical imports, then runtime migration |
| `runtime.plane_beris_edwards -> experimental.plane_shadow_driver` | replace canary bridge before production promotion |

Phase 1 retired `runtime.plane_legacy -> transforms`.  It intentionally kept
the model and configuration facade edges above: directly replacing either one
with an operator/backend import would preserve the cycle in a different form
and violate the target dependency direction.

Key rules are:

1. `core` remains standard-library-only.
2. `geometries` and `planning` remain tensor-free.
3. models do not choose concrete transforms or geometry-specific solvers.
4. backends contain no model physics.
5. applications are composition roots; lower layers do not import them.
6. production code does not acquire new dependencies on `experimental`.
7. unsupported capability combinations fail explicitly; they do not silently
   fall back to another geometry or algorithm.

## GPU and hot-loop constraints

- Resolve strings, registries, capability dispatch, and dependency graphs at
  construction time.
- Use stable tensor shapes, dtypes, layouts, and addresses in the hot path.
- Batch components with compatible basis and storage signatures.
- Keep transient allocation and representation materialization explicit.
- Avoid per-step dictionaries, JSON handling, metadata generation, `.item()`,
  and CPU synchronization.
- Apply `torch.compile` to qualified pointwise islands rather than requiring
  the whole spectral orchestration layer to be compiled.
- Let reference and optimized executors share one scientific plan while using
  different execution policies.

Architectural cleanliness is not allowed to repeat the Stage O separated
runtime regression.  The v0.1.2 data layout and production kernels remain the
performance oracle until a candidate independently qualifies.

## Identity and reproducibility

Every run must distinguish:

1. scientific identity: equations, parameters, conventions, and physical BCs;
2. discretization identity: grid, bases, dtype, dealiasing, integrator, and
   zero-mode choices;
3. execution identity: backend, storage, transform implementation, compile
   policy, and device policy;
4. run identity: initial-condition realization, seed, duration, and output
   schedule.

An optimization may change execution identity without being represented as a
scientific change.  Each layer supplies structured metadata; applications
compose it instead of duplicating implementation descriptions.

## Migration policy

Migration follows a strangler pattern over v0.1.2:

- `legacy_production` remains the supported Plane default and rollback oracle;
- the separated canary remains a numerical oracle and architecture laboratory;
- new declarations are lowered onto the qualified data path one seam at a
  time;
- no third runtime is created merely to bridge the first two;
- candidate qualification and default promotion are separate decisions;
- compatibility facades remain until an announced deprecation period has
  elapsed.

Phase 1 is limited to a mechanical extraction of `pssolver/transforms.py`
behind its existing import facade.  It does not migrate Plane state, rewrite
the timestep, or alter transform mathematics.

## Deferred decisions

The following are deliberately not frozen in v0.2 Phase 0:

- a symbolic equation DSL;
- third-party plugin ABI;
- JAX/CuPy or a universal array backend;
- arbitrary geometry interfaces;
- permanent field packing order or binary checkpoint layout;
- automatic runtime tuning;
- general Newton/Krylov/PETSc integration;
- MPI or multi-GPU decomposition;
- final public APIs for strong anchoring or optimal control.

An interface should have at least two real consumers before it becomes a
stable public commitment.

## Phase 0 decision records

- [ADR 0001](adr/0001-strangler-migration-over-v0-1-2.md): migration over
  the v0.1.2 production oracle.
- [ADR 0002](adr/0002-problem-declarations-and-lowering.md): declarations and
  lowering.
- [ADR 0003](adr/0003-plan-execution-state-and-workspace.md): plan, state, and
  workspace ownership.
- [ADR 0004](adr/0004-geometry-specific-capability-dispatch.md): exact
  geometry capability dispatch.
- [ADR 0005](adr/0005-public-api-and-compatibility-policy.md): API levels and
  compatibility.
- [ADR 0006](adr/0006-identity-provenance-and-qualification.md): identity,
  provenance, and qualification.

Supporting inventories:

- [v0.1.2 API surface](api_surface_v0_1_2.md)
- [v0.1.2 oracle](v0_1_2_oracle.md)
- [Phase 1 transform extraction](phase_1_transform_extraction.md)
- [Phase 1 qualification](phase_1_local_qualification.md)
