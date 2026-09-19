# Phase 2 plan: decompose `PlaneBerisEdwardsRunSpec`

Status: `P2.1_LEAF_COMPONENTS_COMPLETE_STOKES_LAYER_DECISION_PENDING`

Design baseline: Phase 1 closure commit
`fe7b9272c95f0de33cfa3b01b15559611b65786a` on
`next/pssolver-v0.2.0-architecture`.

The machine-readable companion to this plan is
[phase_2_run_spec_inventory.json](phase_2_run_spec_inventory.json).  The
compatibility decision is recorded in
[ADR 0007](adr/0007-run-spec-compatibility-facade.md).

## Objective

Separate scientific, numerical, execution, initialization, and workflow
ownership currently mixed in `PlaneBerisEdwardsRunSpec`, while retaining its
flat supported facade and every v0.1.2 observable contract.

Phase 2 is a configuration and ownership migration.  It does not alter the
Beris--Edwards equations, Plane geometry, boundary conditions, spectral bases,
operation order, timestep, runtime default, output schema, or production GPU
hot path.

## Audited current surface

The facade contains 54 dataclass fields and 17 methods or properties.  Its
fields fall into the following ownership groups:

| Group | Count | Fields |
|---|---:|---|
| Model and Stokes request | 14 | `activity_number`, `parameterization`, `frank_k`, `coefficient_min`, `coefficient_max`, `ldg_a`, `ldg_b`, `ldg_c`, `gamma`, `flow_alignment`, `eta`, `zero_mode_policy`, `friction_mode_fric`, `beta` |
| Domain (geometry and grid) | 6 | `height`, `lx`, `ly`, `nx`, `ny`, `nz` |
| Numerics and time discretization | 7 | `dt`, `dtype`, `dealias_rule`, `projected_transform_execution`, `transform_execution_order`, `spectral_storage`, `spectral_refresh` |
| Initialization | 8 | `seed`, `num_defect_pairs`, `defect_min_separation`, `defect_core_radius`, `background_angle`, `twist_amplitude`, `twist_modes`, `initial_s` |
| Execution | 7 | `device`, `tf32`, `molecular_field_linear_space`, `stress_divergence_sum_space`, `pointwise_execution`, `disable_q_gradient_reuse`, `runtime_path` |
| Workflow/output | 10 | `output_dir`, `steps`, `save_start_step`, `save_interval`, `diagnostic_interval`, `diagnostics`, `save_hydrodynamics`, `dry_run`, `checkpoint_interval`, `restart_from` |
| Boundary aggregate | 1 | `boundaries` |
| Provenance | 1 | `validation_config_sha256` |

The classification above describes intended ownership, not current schema-v1
placement.  For example, schema v1 stores `initial_s` and three execution
selectors under `model`; Phase 2 preserves that serialized placement.  The
single `DomainSpec` owns both lengths and shape in the component graph, while
the grid part of that object belongs to discretization identity.  Likewise,
the zero-mode policy is stored with the Stokes request but is classified as a
discretization/reference-frame choice by ADR 0006; the friction coefficient is
a physical Stokes parameter.

The 17 callable or derived members are:

```text
__post_init__
S_initial
spectral_refresh_mode
spectral_refresh_requested_time
spectral_refresh_requested_steps
spectral_refresh_interval_steps
spectral_refresh_effective_time
shendruk_preset
domain
geometry
numerics
to_metadata
canonical_sha256
runtime_identity_metadata
runtime_identity_sha256
identity_metadata
runtime_selection_metadata
```

The production construction path is CLI parser to factory to flat dataclass.
Direct dataclass construction and the programmatic factory are both supported
surfaces, so neither signature may be replaced in this phase.

## Current consumers and authority hazards

| Consumer | Current responsibility | Main migration hazard |
|---|---|---|
| `applications/plane_beris_edwards.py` | device/dtype resolution, metadata, initialization, early checkpoint gate, runtime and workflow assembly | nearly all fields are read; application also hand-writes production metadata |
| `runtime/plane_legacy.py` | numerical construction and hot-path policy | a changed value or construction order can alter Q/u/p or allocation behavior |
| `runtime/plane_beris_edwards.py` | runtime selection and canary request | mixed-authority and no-fallback checks must remain early |
| `workflows/plane_beris_edwards.py` | scheduling, observation, output, checkpoint/restart | `runtime_identity_sha256` and completion ordering are persistent contracts |
| checkpoint format v1 | restart header and evolved state | old headers must remain readable and same-backend restart byte exact |

Two authority hazards require explicit treatment:

1. `RunSpec.to_metadata()` and the application's hand-written production
   metadata are parallel descriptions.  They must be made to compose from one
   source only after schema-v1 parity is frozen.
2. `boundaries` changes metadata and restart identity, but the qualified
   production equations currently use global boundary constants.  Phase 2
   preserves this historical constructor behavior, records requested and
   effective values separately in its internal view, and does not claim that
   an alternate value is executed.  True lowering is a separately qualified
   feature.

## Reused declarations

Phase 2 does not rebuild types that already express the required meaning:

- `DomainSpec` and `PlaneSlab` own grid/lengths and slab topology;
- `NumericsConfig` owns precision, dealiasing, transform execution, storage,
  and Hermitian-axis intent;
- `BoundarySet` and field-component declarations own physical boundary
  attachment;
- `PlaneFreeSlipBoundaryConditions` remains the schema-v1 compatibility
  aggregate, but is not promoted as a generic boundary API because it also
  contains derived modal parity spaces;
- `ShendrukPlanePreset` owns resolved benchmark coefficients;
- `SpectralRefreshSpec` owns the resolved refresh schedule;
- `IncompressibleStokesSystemSpec`, `TangentialZeroModePolicy`, and
  `PressureGauge` own Stokes viscosity, friction, gauge, and tangential
  zero-mode semantics;
- `BerisEdwardsConstitutiveParameters` is reusable at the lowering boundary,
  but is not by itself a complete production model declaration;
- `ProblemSpec` remains the eventual model/geometry/numerics scientific
  aggregate and is not connected until a tensor-free production
  Beris--Edwards model declaration exists.

`BerisEdwardsPlaneCoupledModel` is not reused as the production declaration:
it is executable, Torch-dependent, and carries separated-canary semantics.

## Provisional component model

The minimum new composition is:

```text
PlaneBerisEdwardsRunComponents
  geometry: PlaneSlab
  boundaries: PlaneFreeSlipBoundaryConditions
  effective_boundaries: PlaneFreeSlipBoundaryConditions
  numerics: NumericsConfig
  physics: PlaneBerisEdwardsPhysicsSpec
    material: BerisEdwardsMaterialRequest
      ldg_a, ldg_b, ldg_c, gamma, flow_alignment, beta
    shendruk_request: ShendrukPlaneParameterRequest
      activity_number, parameterization, raw frank_k, coefficient bounds
    stokes: IncompressibleStokesSystemSpec
      eta, zero-mode policy, effective friction
    requested_friction_mode_fric
  preset: ShendrukPlanePreset
  time_stepping: PlaneTimeSteppingSpec
    dt
    spectral_refresh: SpectralRefreshSpec
  initial_condition: ExtrudedDefectGasInitialConditionSpec
  execution: PlaneBerisEdwardsExecutionSpec
  workflow: PlaneWorkflowSpec
  invocation: PlaneInvocationSpec
```

Only geometry-neutral model-owned tensor-free specifications belong in
`pssolver/models/active_nematics/specifications.py`.  The raw
`ShendrukPlaneParameterRequest` belongs next to its resolver in the existing
`pssolver/presets/shendruk.py`; it retains `activity_number`, parameterization,
raw `frank_k`, and both coefficient bounds even when some are validation-only
inputs.  Plane-only physics composition, execution, workflow, and invocation
value objects belong in
`pssolver/configuration/plane_beris_edwards_components.py` until a second real
consumer justifies a more general public interface.

Ownership rules are:

- `dtype` exists only as `NumericsConfig.precision` in the component view;
- `dt` and refresh are time-discretization identity, not output workflow;
- `steps`, save/checkpoint schedules, output, and restart are workflow/run
  identity;
- `initial_s` belongs to initial conditions despite its legacy metadata key;
- molecular-field space, stress-summation space, pointwise execution,
  Q-gradient reuse, runtime path, device, and TF32 are execution identity;
- viscosity and effective friction are physical Stokes values; pressure gauge
  and tangential zero-mode policy are Stokes-owned declarations with
  discretization/reference-frame identity;
- in `zero_mean` mode, the resolved `IncompressibleStokesSystemSpec` uses
  effective friction zero while the dormant requested
  `friction_mode_fric` is retained losslessly for facade/schema-v1 round-trip;
  in `friction` mode the requested positive value becomes effective friction;
- `dry_run` and `validation_config_sha256` are invocation/provenance, not
  scientific model parameters.

The aggregate is initially produced only by:

```text
decompose_plane_beris_edwards_run_spec(legacy_spec)
```

No nested component becomes a new field of the facade.  That avoids changing
dataclass equality, representation, field order, pickling, or constructor
behavior.

## Cross-component invariants

The composition root, not an arbitrary leaf object, validates relationships
that cross ownership boundaries:

- Shendruk activity resolution uses the wall height and material parameters;
- every `twist_mode` satisfies `1 <= mode < nz`;
- physical-time refresh is an integer multiple of `dt`;
- Hermitian-half storage uses the qualified periodic Hermitian axis;
- truncated projected transforms are compatible with the dealiasing rule;
- resolved `zero_mean` uses effective friction zero, while resolved
  `friction` mode uses the requested positive drag;
- separated-canary restrictions remain explicit, including Q-gradient reuse;
- requested boundaries and the fixed qualified effective boundaries are never
  conflated; alternate requests remain historical metadata-only behavior in
  Phase 2 and are not advertised as executable boundary support.

Validation timing, exception types, and stable error substrings are part of
the compatibility characterization.  Phase 2 does not move every factory
validation into the facade's `__post_init__`.

## Existing identity contracts

`canonical_sha256()` hashes sorted, compact JSON from the complete resolved
schema-v1 `to_metadata()`.  It is a full run-request identity and currently
includes output paths, workflow, initial state, runtime controls, and restart
paths.

`runtime_identity_sha256()` binds the same-backend numerical/restart contract.
It excludes output policy, run duration, save/diagnostic schedule, validation
provenance, dry-run, restart path, and all fresh-run initialization fields.
It includes runtime path, device request, TF32, `dt`, refresh, numerical
selectors, boundaries, resolved preset, model coefficients, and zero-mode
choices.

Phase 2 preserves these facts, including the following non-obvious behavior:

- changing a raw, ineffective `frank_k` in `paper-window` mode does not change
  the resolved canonical hash;
- for an already-valid `paper-window` request, `coefficient_max` is a
  validation-bound input and does not enter resolved metadata; the request
  object must nevertheless retain it;
- changing `frank_k` in `fixed-k` mode changes resolved identity, while both
  coefficient bounds are retained raw validation inputs rather than resolved
  identity values;
- path strings are not resolved or normalized;
- changing workflow or initial-condition values can change the canonical hash
  without changing the runtime identity.

Known identity debts are documented rather than repaired here: requested
versus effective device identity, dormant friction values outside friction
mode, diagnostics/cache interactions, and the absence of separate scientific,
discretization, execution, and run hashes in schema v1.

## Prospective commit sequence

### P2.0: freeze compatibility oracles

Add characterization only; do not change production code.

- exact 54-field name/order/default inventory;
- dataclass field name/order/type/default/default-factory/init/repr/compare,
  frozen/slots, `__match_args__`, equality/hash/repr, pickle global path and
  round-trip;
- all 17 existing property/method names, return types, and representative
  values;
- class and factory parameter order/kind/default/annotation signatures;
- parser option strings, `dest`, type, `nargs`, action, required state,
  defaults, choices, and mutually exclusive groups;
- `pssolver.configuration` export inventory;
- schema-v1 golden metadata and canonical JSON/hash cases;
- exhaustive per-field canonical/runtime identity partition;
- raw-versus-resolved preset behavior, dormant requested friction, and
  path-string behavior;
- old checkpoint header/runtime-identity fixtures;
- current alternate-boundary mismatch characterization.

At minimum, golden cases cover defaults, float64 with refresh disabled,
step-based refresh, friction mode, numerical rollback policies, separated
canary, and a workflow/restart-only variation.

P2.0 was completed on 2026-09-18 without changing production code.  Its
static compatibility evidence is:

- `tests/fixtures/configuration/plane_run_spec_api_v1.json`;
- `tests/fixtures/configuration/plane_run_spec_v1_cases.json`;
- `tests/fixtures/configuration/plane_checkpoint_v1_header/checkpoint.json`;
- `tests/test_phase2_run_spec_compatibility.py`.

The focused P2.0 suite contains 75 passing tests.  The complete local CPU
suite contains 1228 passing tests and 8 passing subtests.  These oracles are
the fail-closed baseline for P2.1 and later Phase 2 commits; they are not a
schema-v2 proposal and do not endorse the historical requested/effective
boundary mismatch as a future design.

### P2.1: add disconnected immutable components

Add the two provisional modules, the raw request beside the existing Shendruk
preset resolver, and their local validation tests.  Geometry and benchmark
resolution must not leak into the geometry-neutral model specification.  Do
not change the factory, parser, facade, serializer, consumer, runtime, or
source provenance list.

The dependency-safe leaf portion of P2.1 was completed on 2026-09-18.  It
adds:

- `BerisEdwardsMaterialRequest` and
  `ExtrudedDefectGasInitialConditionSpec` in the stdlib-only model
  specification module;
- lossless raw `ShendrukPlaneParameterRequest` beside the existing resolver;
- `PlaneTimeSteppingSpec`, `PlaneBerisEdwardsExecutionSpec`,
  `PlaneWorkflowSpec`, and `PlaneInvocationSpec` as disconnected Plane
  configuration leaves.

These names remain provisional direct-module imports.  They are not exported
from a stable package surface, connected to the facade, or consumed by a
runtime.  The two newly created modules are not added to the production source
inventory; the pre-existing `presets/shendruk.py` entry remains present and
therefore transparently records the raw-request addition in implementation
provenance without changing the inventory itself.

The physics aggregate and final run-components aggregate remain intentionally
deferred.  The accepted sketch requires
`PlaneBerisEdwardsPhysicsSpec.stokes` to reuse
`IncompressibleStokesSystemSpec`, but that declaration currently lives in
`pssolver.execution`, while the dependency ratchet deliberately forbids
`configuration -> execution` and this phase forbids broadening the allowlist.
P2.1 does not hide that edge through `Any`, a string-only annotation, dynamic
import, re-export indirection, or a duplicated Plane Stokes declaration.  A
separate architecture decision must either place the tensor-free physical
Stokes declaration in an admissible declaration layer or explicitly revise
the layer direction before the physics aggregate is added.

The focused P2.1 leaf suite contains 65 passing tests.  The combined P2.0,
P2.1, import-boundary, and O.1 gate contains 157 passing tests.  The complete
local CPU suite contains 1293 passing tests and 8 passing subtests; clean
wheel import and console-entry smokes also pass.

### P2.2: add pure decomposition and parity adapter

Implement decomposition from the facade to components.  Prove that every
legacy field can be recovered and that schema-v1 metadata, canonical hash,
runtime identity, and runtime-selection metadata are unchanged.  No runtime
consumer is migrated.  The round-trip includes raw preset inputs, dormant
requested friction, requested boundaries, and the separately recorded
qualified effective boundaries.

### P2.3: delegate existing derived views

Let `domain`, `geometry`, `numerics`, and `shendruk_preset` delegate to the
component composition while preserving returned types and values.  Keep all
flat fields and compatibility properties.

### P2.4: freeze an explicit schema-v1 serializer adapter

Make the compatibility serializer an explicit authority without changing a
key, value, nesting level, list/tuple conversion, path spelling, JSON bytes,
or hash.  Component metadata remains internal.

### P2.5: migrate consumers one per commit

Use this order:

1. `runtime/plane_legacy.py`;
2. `runtime/plane_beris_edwards.py`;
3. `workflows/plane_beris_edwards.py`;
4. `applications/plane_beris_edwards.py`.

The application comes last because it combines initialization, checkpoint
gating, runtime construction, workflow construction, and a second metadata
description.

### P2.6: retire the exact configuration import debts

Remove:

```text
configuration.plane_beris_edwards -> plane
configuration.plane_beris_edwards -> transforms
```

Qualified numerical defaults move to tensor-free specification policy.  The
configuration layer must not replace these edges with dependencies on
backends, operators, solvers, runtimes, applications, or experimental code.

### P2.7: provenance and Phase 2 closure

When component files begin determining production meaning, add them to the
implementation source inventory.  This legitimately changes implementation
provenance, not scientific configuration or numerical output.  Complete local
qualification and, after the P2.5 consumer migrations, one final balanced
R128/R320 H100 non-regression task.  No H100 job is required per commit.

## Per-commit gates

Every implementation commit requires:

- `git diff --check`;
- focused component/facade/serializer tests;
- import-boundary tests with no broadened allowlist;
- exact schema-v1 metadata and hash fixtures;
- exact dry-run configuration identity;
- complete CPU suite;
- clean wheel/import and console-entry smoke when package files change;
- no change to runtime default or fallback behavior.

Consumer migrations additionally require a short CPU trajectory and restart
gate.  Numerical outputs must be byte-identical because no floating-point
operation is intentionally reordered.

## Phase-completion gates

Phase 2 is complete only when all of the following hold:

1. all 54 flat attributes and the supported constructor/factory signatures
   remain available;
2. dataclass field order and flags, `__match_args__`, equality/hash/repr,
   pickle path, all 17 members, and representative return values remain
   compatible;
3. factory parameter order/kind/default/annotations and validation exception
   type, timing, and stable error substrings remain compatible;
4. CLI option strings, `dest`, type, `nargs`, actions, required state,
   defaults, choices, mutually exclusive groups, error boundaries, and
   dry-run semantics are unchanged;
5. schema version remains 1;
6. every golden `to_metadata()` document, canonical JSON byte string, and
   canonical SHA-256 is unchanged;
7. every golden runtime identity document and SHA-256 is unchanged;
8. old format-v1 checkpoint headers remain readable;
9. same-backend split restart remains byte-exact and cross-runtime mismatch
   still fails before output or array loading;
10. callable API and CLI short trajectories produce byte-identical Q/u/p,
    initial-condition artifacts (`Q2D_initial.npy`, `Q2D_defects.csv`),
    `raw_q_sha256`, `projected_q_sha256`, `retained_q_modes`,
    `retained_normal_velocity_modes`, and the same output filenames, shapes,
    dtypes, and final `COMPLETE` ordering;
11. schema-v1 metadata values are canonically equal except for a prospectively
    frozen allowlist of implementation-provenance source files/hashes and
    environmental measurements;
12. dry-run emits the same JSON contract without creating an output directory
    or allocating runtime tensors;
13. complete CPU, wheel, entry-point, and dependency-boundary gates pass;
14. no transform, kernel, integrator, timestep order, production default, or
    qualified boundary condition changes;
15. the two exact configuration dependency debts are removed, not renamed;
16. no provisional component is promoted to a stable root public API;
17. one final balanced R128/R320 H100 non-regression task passes after all
    P2.5 consumer migrations.

Local qualification is sufficient for individual commits.  The single final
H100 task is a Phase 2 closure gate, not authorization for a production
default change.

## Explicitly deferred

- schema-v2 metadata or checkpoint format;
- new scientific/discretization/execution/run hashes replacing schema v1;
- path normalization;
- removal of direct facade construction;
- renaming CLI options or metadata keys;
- moving all validation into dataclass construction;
- generic cross-model run specifications;
- using the separated-canary executable model as production declaration;
- alternate Plane boundaries, strong anchoring, and new DCT/DST lowering;
- state/workspace/timestep architecture migration;
- performance optimization.
