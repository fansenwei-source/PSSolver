# PSSolver architecture

Status: first edition, written for the 0.5 unified-engine line (draft r2). The
implementation lands step by step during 0.5, and this file is updated in the
same commits. Every design note for a PSSolver change starts by citing the
section of this file it touches. A change that alters a layer boundary updates
this file in the same commit.

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

## 2. Current layering (v0.4.0) and where each part goes

v0.4.0 runs four hand-wired combinations on one old engine:
- `solver.SpectralSolver`, `PDEmodel.PDEModel` and `Field.Fields`, stepped by
  `integrator.SemiImplicitEulerIntegrator` and its projected subclasses;
- each combination has its own compiler, application, runtime, workflow and
  checkpoint module.

| v0.4.0 package or module | Role today | 0.5 destination |
|---|---|---|
| `api/` | public `Simulation`, `run_simulation`, results, capability catalog (hand-maintained) | kept; catalog derived from the capability table; new `Simulation` from the S-28 switch |
| `configuration/` | declaration types, per-combination compilers, lowering, registry | removed; validation moves to `planning/`; the declaration types needed by functional API 1.1 are frozen into `functional/legacy_declarations/` |
| `planning/` | lowering plans (mostly hashed, not executed), old assembler attempt | becomes the assembler and the executed `Plan` |
| `execution/` | `RuntimeState`, workspace (used); canary-only contracts | state and workspace merge into `engine/`; the rest is removed |
| `applications/`, `workflows/`, `runtime/` (per combination), `adapters/`, `systems/` | per-combination wiring, time loops, checkpoints | removed; replaced by `planning/`, `engine/`, `runtime/runner.py`, `io/` |
| `solver.py`, `PDEmodel.py`, `Field.py`, `integrator.py`, `plane.py`, `channel.py`, `transforms.py` | old engine and facades | removed at S-28; `engine/` replaces them |
| `integrators/` | state-backed Euler step program; scalar SBDF2 reference | IMEX scheme protocol and Euler; the reference stays as a test oracle |
| `models/active_nematics/` | Beris–Edwards kernels (`beris_edwards.py`, `q_tensor.py` are frozen in 0.5), Stokes coupling | kept; wrapped by the preset; called through `engine.FieldView` and the shared projector |
| `backends/`, `operators/`, `linear_solvers/` | transforms, projector, spectral operators, Stokes solvers (four Channel PCG loops) | kept; the Channel solver gains `solve_transpose` and becomes the single PCG |
| `functional/` | functional API 1.1 with its own step rewrite | kept as a facade over the engine; inputs and identity unchanged |
| `io/` | checkpoint primitives, identity, legacy identity registry | kept; one new family `engine_state`, plus `io/legacy/` readers and the compatibility map |
| `experimental/` | `stage_*` process artifacts, canary, boundary pilots | process artifacts move to `qualification/`; the canary is removed; only boundary pilots remain |

## 3. Target layers and dependency direction (end of 0.5)

S-24 status: the legacy active-force Channel application and the separated
canary connection are retired. The three complete-stress combinations remain
on their existing engine. Thirty-five frozen stage artifacts now live outside
the package in `qualification/stages/`; only the seven independent A-group
plans have test-only old-name loading. Historical wiring audits run against a
separate v0.4.0 Git checkout. Neither loader nor historical implementation is
distributed. The shared pressure-policy declaration lives privately alongside
the canonical Stokes solver. State/workspace, the old assembler and private
lifting wiring remain for their explicitly scheduled later steps.

```
public       pssolver (top-level exports), api/     Simulation, run_simulation, results
components   models/ presets/ geometries/ boundaries/ integrators/
assembly     planning/                              assemble() -> CompiledSimulation(plan)
plan         planning/plan.py                       immutable, tensor-free, hashable
engine       engine/                                State, step program, operators, constraints
numerics     backends/ operators/ linear_solvers/   transforms, projector, Stokes solvers
run          runtime/                               the single Runner
persistence  io/                                    checkpoint format, legacy readers, identity
facades      functional/ cli/                       functional API 1.1 (+ its legacy declarations), CLI
vocabulary   core/                                  canonical hashing, identifiers, errors, specs
```

Dependencies point downward only. `tools/check_imports.py` checks the module
import contracts below (rules in `pyproject.toml`,
`[tool.pssolver.import-contracts]`):

| Package | Must not import |
|---|---|
| `core`, `planning` | `torch`; `engine`, `runtime`, `io`, `api`, `functional`, `cli`; the implementation modules of `models` and `presets` |
| `backends`, `operators`, `linear_solvers` | `api`, `runtime`, `io`, `functional`, `cli`, `planning` (except `planning.plan` types) |
| `engine` | `api`, `runtime`, `io`, `functional`, `cli` |
| `runtime` | `api`, `functional`, `cli` |
| `io` | `api`, `runtime`, `functional`, `cli`; from `engine`, only the `State` type |
| `models`, `presets`, `geometries`, `boundaries`, `integrators` | `engine` internals, `runtime`, `io`, `api` |
| everything in the package | `experimental`; the moved `pssolver.experimental.stage_*` names |

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

`planning.plan.Plan` is a frozen dataclass without tensors. It records:
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
  action);
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

Tiers 2 and 3 are reference implementations in 0.5.

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

`integrators` defines an IMEX scheme protocol:
- implicit denominator;
- explicit update;
- history length;
- startup policy.

0.5 ships semi-implicit Euler, whose operation order is part of the numerical
contract. History is part of `State` and of checkpoints.

### 4.4 Runner and checkpoints

`runtime.Runner` is the only time loop. It owns:
- output, observation and checkpoint cadences;
- restart preflight, which rejects before creating output directories or
  building the engine;
- run metadata, through a single builder.

`io` writes one checkpoint family, `engine_state`: a directory of `.npy` files
plus `checkpoint.json`, with a metadata hash and a layered identity.

`io/legacy/` contains:
- read-only importers for older formats;
- the frozen v0.4 compatibility-identity function;
- the compatibility map.

A legacy checkpoint is resumed only if all of the following hold:
- its identity type is whitelisted in the map;
- its disposition allows exact restart;
- its stored identity equals, byte for byte, the identity the frozen v0.4
  function computes for the current run.

Everything else is rejected before any tensor is read.

## 5. Status of public surfaces

| Surface | Status |
|---|---|
| `pssolver.Simulation`, `run_simulation`, `compile_simulation`, results, capability catalog | stable within 0.x minor versions. During 0.5 development the v0.4 `Simulation` stays the only public front end until the S-28 switch; migrated combinations reach the engine through the 1.1 translator |
| `pssolver.engine` (`Engine`, `State`, model terms), `planning.assemble` | internal until the S-28 switch, then public and provisional until 1.0 |
| `pssolver.functional.api` (functional API 1.1) | stable; inputs (v0.4 `SimulationSpec`) and `identity()` unchanged; kept until its consumers move to 2.0 |
| `pssolver.functional.legacy_declarations` | 1.1 compatibility surface only: v0.4 declaration types and their translator to components. Not exported at top level; cannot be passed to `run_simulation`; removed together with 1.1 |
| `engine.FieldView`, projector interface for frozen kernels | compatibility surface; new models must not use it |
| Plane command line `pssolver-plane-beris-edwards` | stable command name and numerical options; `--runtime-path` removed in 0.5 |
| `experimental/` | not supported; contains only boundary pilots awaiting the boundary work. Each module docstring states its promotion-or-removal condition |
| `control/` | not shipped in the wheel |

## 6. Exactly one way to do each thing

Counts are of modules touched; the target is at most three.

| Task | Where and how |
|---|---|
| Add a model | Write fields, parameters and terms in `models/` (see `docs/adding_a_model.md` and `examples/model_template/`). Derive manufactured solutions with SymPy in `tests/mms/`. Register the preset in `presets/__init__.py`. Two modules plus tests; `engine/` untouched. |
| Add a geometry built from existing axis kinds | The geometry component in `geometries/`. Basis rules are per axis kind and boundary condition, so they need no change. If the geometry creates a new constraint basis signature: add one solver module in `linear_solvers/` and one entry in `planning/capabilities.py`. At most three modules. |
| Add a new axis kind or basis | Not a geometry change. Basis rules in `planning/bases.py` plus the transform in `backends/`. Needs its own design. |
| Add a boundary condition | Physical declaration in `boundaries/`; basis or lifting rule in `planning/`; lifting in `engine/` (from the boundary work onward). |
| Add a constraint solver | Implement the `ConstraintSolver` protocol in `linear_solvers/`, with `solve_transpose` and a dot-product test. Add one entry in `planning/capabilities.py`. Each entry is a torch-free record: key, implementation reference and adjoint mode. A test validates every reference. |
| Add an optimized implementation | Register it against its reference operator with an equivalence record: bitwise, or a tolerance defined before measurement and approved where a project rule requires it. Select it through `execution` options. |
| Add a time scheme | Implement the IMEX protocol in `integrators/`; prove the order of time convergence. |
| Add a run-metadata field | The single metadata builder in `runtime/`. |
| Add a qualification or diagnostic script | `qualification/` at the repository root (not in the package, wheel or sdist). Historical import paths are served only by `qualification/_legacy_imports.py`, for tests and the qualification launcher. |
| Reject an unsupported request | In `planning/`, before allocation. Either a mathematical reason (which operator, basis or solver is missing, or which data conflict), or a scope restriction naming the decision and the planned version. |

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

The numbers are tracked as trends, not gates. A regression is explained in the
release design.
# S-25 Periodic draft boundary

The Periodic production and functional paths share `engine.Engine` and caller-owned
`engine.State`. The torch-free `planning.periodic` produces a Plan. The v0.4 RunSpec,
identity, upgrade and embedded-identity validation remain in
`configuration.legacy_periodic` until S-28 (D-0051 A). `io.legacy.periodic_workflow`
contains only the frozen carrier and its read/write operations. `runtime.legacy_entry`
owns the declaration-to-Engine entrance; `runtime.runner` owns scheduling and output.
Plane and Channel remain on their existing paths. The protected physics and Plane
solver files are unchanged. No old Periodic adapter shell remains.

Frozen benchmark consumers of deleted Periodic modules are not repaired here. Their
original tests execute in the v0.4 historical checkout; exact non-importable chains
are recorded in `qualification/s25_expected_unimportable.json` for S-30. S-24's test
audit ends at aee8289; S-25 changes are audited independently by
`qualification/s25_test_disposition.json`. This section describes an unqualified draft,
not a claim that T1–T25 or K2 has passed.
