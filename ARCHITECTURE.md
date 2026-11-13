# PSSolver architecture

This is the architecture of the 0.5 unified-engine line. Every design note for
a PSSolver change starts by citing the section it touches. A change that alters
a layer boundary updates this file in the same commit.

## 1. What PSSolver is

PSSolver is a GPU-first, differentiable pseudo-spectral framework for continuum
field theories on regular tensor-product domains (periodic boxes, slabs, ducts).
It is not a general PDE package. Arbitrary meshes, finite elements, AMR,
multi-GPU and a symbolic DSL are out of scope for now.

One engine executes every supported problem. A problem is assembled from
components: model, geometry, physical boundary conditions, time scheme,
numerics, initial condition and output. Presets are recipes of components. Fast
paths are optimized implementations of the same mathematics, not separate
drivers.

## 2. From declaration to execution

`pssolver.api.runner.compile_simulation` validates a `Simulation` and calls
`pssolver.planning.assemble.assemble`. The resulting immutable
`pssolver.planning.engine_plan.Plan` is the executable, tensor-free contract.
`pssolver.engine.engine.Engine` materializes it on the requested device.
`pssolver.runtime.runner.Runner` drives the engine and publishes observations,
diagnostics and checkpoints. `pssolver.io.engine_state` owns the native
checkpoint format. Periodic, Plane and Channel use this same route.

`pssolver.functional.api` keeps its 1.1 declaration identity and uses the same
engine. Its Channel step uses zero-start pressure and a custom transpose VJP;
the unrolled reference is not a production solver. The frozen legacy
declarations preserve released input and checkpoint contracts without
reintroducing the old execution drivers.

For the historical v0.4 frontend and scripts, use the
[migration guide](docs/migration_0_4_to_0_5.md).

## 3. Layers and dependency direction

```
public       pssolver (top-level exports), api/     Simulation, run_simulation, results
components   models/ presets/ geometries/ boundaries/ integrators/
assembly     planning/                              assemble() -> CompiledSimulation(plan)
plan         planning/engine_plan.py                immutable, tensor-free, hashable
engine       engine/                                State, step program, operators, constraints
numerics     backends/ operators/ linear_solvers/   transforms, projector, Stokes solvers
run          runtime/                               the single Runner
persistence  io/                                    checkpoint format, legacy readers, identity
facades      functional/ cli/                       functional API 1.1 (+ its legacy declarations), CLI
vocabulary   core/                                  canonical hashing, identifiers, errors, specs
```

`qualification/s30_guard.py` checks the full import contract against
`[tool.pssolver.architecture-contracts]` in `pyproject.toml`. The existing
`tools/check_imports.py` checks the original subset. The full contract covers
static and literal-resolved dynamic imports. Its cycle check covers explicit
module-initialization imports; local and type-checking-only imports are still
dependencies, but not cycle edges.

| Package | Must not import |
|---|---|
| `core`, `planning` | `torch`, `engine`, `runtime`, `io`, `api`, `functional`, `cli`, `models`, `presets` |
| `backends`, `operators`, `linear_solvers` | `api`, `runtime`, `io`, `functional`, `cli` |
| `engine` | `api`, `runtime`, `io`, `functional`, `cli` |
| `runtime` | `api`, `functional`, `cli` |
| `io` | `api`, `runtime`, `functional`, `cli`, `engine` |
| `models`, `presets`, `geometries`, `boundaries`, `integrators` | `engine`, `runtime`, `io`, `api` |
| `outside_experimental` | `experimental` |

The only explicitly registered edges for this table are:

- `pssolver.io.engine_state` imports `State` from `pssolver.engine.state`.
- `pssolver.planning.assemble.AssembledProblem.engine` imports `Engine` from
  `pssolver.engine.engine` only when called. Importing planning does not load
  Engine or Torch.
- `pssolver.integrators.reference_checkpoint` is a reference-test I/O adapter.
  It imports `checkpoint_directory_transaction` from `pssolver.io`, and
  `canonical_indented_json_bytes`, `indented_identity_sha256`, `raw_file_sha256`,
  `read_reference_tensor_record`, `sha256_bytes` and `write_reference_tensor_record`
  from `pssolver.io._checkpoint_artifacts`.
- `pssolver.operators.__getattr__` uses
  `pssolver.core._compatibility.resolve_deprecated_facade_name` to lazily resolve
  the 13 exact exports in `[tool.pssolver.architecture-facades]` from
  `pssolver.experimental.boundary_pilots`. Name access preserves object identity
  and emits the existing deprecation warning. This is not a general dependency
  on experimental modules.

The frozen v0.4 declarations in `pssolver.functional.legacy_declarations` are
reported as historical dependencies and are not rewritten.

How `planning` stays torch-free while presets wrap torch kernels:
- Component objects are created by the caller and passed into `assemble()`.
- `planning` only reads their declarative attributes: fields, parameters, term
  metadata, and kernel references held as opaque objects.
- `planning` never calls a kernel and never imports a component's
  implementation module.
- The kernels run only inside `engine`.

`assemble()` allocates no tensors. Tests check this with a tensor-allocation
counter wrapped around `assemble()`, including every rejection path. That is how
"reject before allocation" is checked: there is no blanket rule such as "only
the engine allocates", which could not be checked.

## 4. Core contracts

### 4.1 Plan

`planning.engine_plan.Plan` is a frozen dataclass without tensors. It records:
- per-field, per-component, per-axis bases;
- packed transform groups;
- linear-operator tiers;
- explicit terms with their derivative requests;
- constraint solves;
- dealiasing;
- the time scheme;
- parameters, with `controllable` flags;
- boundary-data slots (empty in 0.5);
- outputs.

`assemble()` computes the Plan identity hash exactly once. Run metadata and
`engine_state` checkpoints refer to that hash. Two identities keep their v0.4
definitions:
- functional API 1.1 `identity()`;
- the legacy checkpoint compatibility identity (section 4.4).

### 4.2 Engine

- `Engine(plan, device=, dtype=, batch=)` materializes operators once.
- `engine.prepare_constraints(state)` performs only the constraint update and
  sets `constrained_current`.
- `engine.step(state, controls)` is a pure function.
- `engine.step_(state, controls)` is the in-place execution of the same step
  program: one stage list, whose stages accept optional `out=` buffers.
- The two modes are bitwise identical. Never write a second copy of a step.

Hot-loop rules:
- The hot loop executes the packed transform groups and storage layout of v0.4.
- It adds no resident buffers.
- Component structure is metadata only.

`State` holds:
- spectral and spatial fields, with layout `(field, batch, *grid)`;
- integrator history;
- constraint-solver state (for example a warm-start guess and its operator
  action). Channel keeps that action in its original working scale, with a
  Python integer exponent per member. Replacing a guess must replace the
  matching action and exponent or clear both; the Engine never updates the
  solver tensors in place;
- the step counter, refresh phase and `constrained_current` flag.

Everything that influences the next step is in `State`; diagnostics are not.

**Observation contract.** Observations and checkpoints happen after the
constraint update and before the explicit right-hand side. The Runner calls
`prepare_constraints`, observes, then calls `step`, which reuses the constraint
solution. `observe` refuses a state whose constraints are not current.

**Linear operators** come in three tiers:
1. diagonal per mode (production);
2. small dense blocks per mode;
3. general matrix-free solves.

All three tiers use the same `advance → linear.solve` call. The production
diagonal implementation keeps the original Euler denominator and division
order. Tier 2 is a private modal-block reference; tier 3 defines a private
matrix-free protocol with caller-owned diagnostics. Its CG example lives in
tests, not in the library. Public assembly selects only the diagonal tier.

Generic model terms share a table of physical derivatives keyed by field, axis
orders and basis. The table is owned by one explicit evaluation and emptied
even when evaluation raises. The protected Beris–Edwards paths retain their
separate, existing Q-gradient reuse contract.

**Constraint solvers** are keyed by `(operator kind, basis signature)`. Each:
- provides `solve` and `solve_transpose`;
- declares its adjoint mode (`autograd` or `custom_transpose`);
- reduces and converges per batch member;
- follows a warm-start policy (`previous` or `zero`) recorded in the Plan.

`batch` is a first-class axis everywhere. Public entry points fix `batch = 1`
in 0.5.

**Dealiasing.**
- The Plan fixes the rule.
- The engine builds one projector and applies the end-of-step and initial
  projections itself.
- Kernels that predate model terms receive that same projector object and
  project internally, as in v0.4.

**Compatibility surfaces for kernels that predate model terms.**
- `engine.FieldView` reproduces exactly the subset of the v0.4 `Fields` API
  those kernels use.
- The projector exposes `forward_transform`, `inverse_transform` and
  `inverse_transform_full`, unchanged.
- New models must not use either.

### 4.3 Time schemes

`engine.imex` defines the engine's IMEX scheme protocol:
- implicit denominator;
- implicit shift;
- explicit update;
- history length;
- startup policy.

0.5 ships semi-implicit Euler, whose operation order is part of the numerical
contract. History is part of `State` and of checkpoints. Test-only SBDF2 uses
the same step program and starts with Euler; it is not a public integrator.

### 4.4 Runner and checkpoints

`runtime.Runner` is the only time loop. It owns:
- output, observation and checkpoint cadences;
- restart preflight, which rejects before creating output directories or
  building the engine;
- run metadata, through a single builder.

`io` writes one checkpoint family, `engine_state`: a directory of `.npy` files
plus `checkpoint.json`, with a metadata hash and a layered identity.
Public readers reject checkpoints with integrator history or an explicit
non-Euler scheme before touching any tensor payload.

`pssolver.io.legacy.periodic_workflow` contains the legacy Periodic checkpoint
carrier and reader. The frozen v0.4 identity and declaration rules live under
`pssolver.functional.legacy_declarations`, not in the Engine.

A legacy checkpoint is resumed only if all of the following hold:
- its identity type is whitelisted in the map;
- its disposition allows exact restart;
- its stored identity equals, byte for byte, the identity the frozen v0.4
  function computes for the current run.

Everything else is rejected before any tensor is read.

## 5. Status of public surfaces

| Surface | Status |
|---|---|
| `pssolver.Simulation`, `run_simulation`, `compile_simulation`, results, capability catalog | The 0.5 public front end assembles once and hands that Plan to the common Runner; the catalog derives its capabilities from assembly |
| `pssolver.engine` (`Engine`, `State`, model terms), `planning.assemble` | Public and provisional until 1.0 |
| `pssolver.functional.api` (functional API 1.1) | stable; inputs (v0.4 `SimulationSpec`) and `identity()` unchanged; kept until its consumers move to 2.0 |
| `pssolver.functional.legacy_declarations` | 1.1 compatibility surface. Also used privately to authenticate and resume qualified v0.4 checkpoints. Not exported at top level; cannot be passed to `run_simulation`; the declaration surface retires with 1.1 |
| `engine.FieldView`, projector interface for frozen kernels | compatibility surface; new models must not use it |
| Plane command line `pssolver-plane-beris-edwards` | stable command name and numerical options; `--runtime-path` removed in 0.5 |
| `experimental/` | not supported; contains only boundary pilots awaiting the boundary work. Each module docstring states its promotion-or-removal condition |
| `control/` | not shipped in the wheel |

## 6. Exactly one way to do each thing

Counts are of modules touched; the target is at most three.

| Task | Where and how |
|---|---|
| Add a model | Follow `docs/adding_a_model.md` and `examples/model_template/`; declare `pssolver.core.terms.PackageTerm` records and provide a preset such as `pssolver.presets.shendruk.resolve_shendruk_plane_preset`. Derive manufactured solutions independently. The step program stays untouched. |
| Add a geometry built from existing axis kinds | Follow `pssolver.geometries.tensor_product.PeriodicBox`. If it introduces a new constraint signature, register it using `pssolver.planning.capabilities.basis_signature` and `pssolver.planning.capabilities.Capability`. |
| Add a new axis kind or basis | Needs its own design. `pssolver.planning.assemble._validate_basis` owns basis rejection, and `pssolver.engine.groups` owns transform-group construction. |
| Add a boundary condition | Follow the physical declarations in `pssolver.boundaries.neumann_q`; basis validation stays in `pssolver.planning.assemble._validate_basis`. Lifting and unqualified boundary pilots are not added as production options here. |
| Add a constraint solver | Follow `pssolver.linear_solvers.stokes.channel_no_slip.ChannelNoSlipModalStokesSolver.solve_transpose` for the adjoint and add a dot-product test. Register the solver with a torch-free `pssolver.planning.capabilities.Capability`. |
| Add an optimized implementation | Use the existing selection in `pssolver.planning.assemble.assemble` and the declarations in `pssolver.planning.capabilities`. Bind the reference and equivalence evidence before making an implementation selectable. |
| Add a time scheme | Implement `pssolver.engine.imex.ImexScheme` and prove time convergence order. |
| Add a run-metadata field | Extend `pssolver.runtime.simulation_run.build_run_metadata`. |
| Add a qualification or diagnostic script | Use `qualification/`, outside the distribution, and run it through `qualification/run.py`. `qualification/_legacy_imports.py` serves the approved historical imports for development tests only. |
| Reject an unsupported request | Use `pssolver.planning.assemble.AssemblyRejection` before allocation. Identify the mathematical conflict or the governing scope decision. |

## 7. Verification contract

- **Migration and refactoring:** old and new paths are bitwise identical on CPU
  in float64 and float32 for N steps, including restart and observation timing.
  Any tolerance needs an explicit project decision.
- **Optimized implementations:** bitwise, or within a tolerance defined before
  measuring.
- **Manufactured solutions:** derived with SymPy outside the package. They are
  independent of production kernels; deliberate duplication is allowed.
- **Integrators:** each has a time-convergence-order test.
- **Constraint adjoints:** dot-product tests, finite-difference checks and
  Taylor tests.
- **Tests and gates:**
  - no new skip or xfail;
  - a deleted test names the contract it protected and where that contract is
    now covered;
  - tests that need a prerequisite (baseline fingerprint, v0.4.0 source tree)
    are selected by category, not skipped or deselected.

## 8. Toward 1.0

| Version | Step toward the target form |
|---|---|
| 0.5 | One engine, one assembler, one runner. Functional API 1.1 runs on the engine with unchanged inputs and identity. Process artifacts leave the package. |
| 0.6 | Functional API 2.0 (batch, any declared parameter as control, possibly boundary data). Boundaries and anchoring through the assembler. Frozen kernels move from `FieldView` to model terms once their consumers have migrated. |
| 0.7 | Production SBDF2/CNAB2. A second real model built from the template. Noise. Production-size adjoints. |
| 1.0 | Stable API, checkpoint and metadata schemas. Functional API 1.1, its legacy declarations and other facades past their removal version are deleted. Documentation and CI. |

## 9. Structure numbers

`tools/structure_metrics.py` reports, once per release:
- package size, with `functional/legacy_declarations` reported separately;
- experimental share;
- duplication;
- functions over 150 lines;
- orchestration-to-numerics ratio;
- import cycles;
- unimported modules.

The numbers are tracked as trends, not gates. The release design explains
regressions and records public documentation coverage and functions over 80
lines alongside these metrics.
