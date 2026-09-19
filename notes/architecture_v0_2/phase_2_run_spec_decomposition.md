# Phase 2 plan: decompose `PlaneBerisEdwardsRunSpec`

Status: `P2.5_IN_PROGRESS_TWO_RUNTIME_CONSUMERS_COMPLETE`

Design baseline: Phase 1 closure commit
`fe7b9272c95f0de33cfa3b01b15559611b65786a` on
`next/pssolver-v0.2.0-architecture`.

The machine-readable companion to this plan is
[phase_2_run_spec_inventory.json](phase_2_run_spec_inventory.json).  The
compatibility decision is recorded in
[ADR 0007](adr/0007-run-spec-compatibility-facade.md).  The declaration-layer
decision that unblocks the Stokes component is recorded in
[ADR 0008](adr/0008-tensor-free-system-declarations.md).

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
  `PressureGauge` in `pssolver.systems.stokes` own Stokes viscosity,
  friction, gauge, and tangential zero-mode semantics;
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
value objects are physically defined in the dependency-neutral provisional
`pssolver/configuration/plane_beris_edwards_component_graph.py` module.  Their
P2.1 nominal/direct-import path remains
`pssolver/configuration/plane_beris_edwards_components.py`, which is now the
upper compatibility adapter.  Neither path becomes a stable package-root API
until a second real consumer justifies a more general interface.

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

The aggregate portion of P2.1 was completed on 2026-09-19 after ADR 0008's
declaration-layer prerequisite passed.  It adds:

- `PlaneBerisEdwardsPhysicsSpec`, which composes the material request, raw
  Shendruk request, canonical Stokes request, and separately retained raw
  friction-mode value;
- `PlaneBerisEdwardsRunComponents`, which owns the 11-part disconnected
  component graph and validates Plane topology, requested/effective boundary
  separation, preset resolution, twist modes, refresh consistency, the
  qualified Hermitian axis, and the separated-canary cache restriction.

The aggregate's nested metadata is explicitly provisional and is not a
schema-v1 serializer, identity source, or production metadata authority.  The
new objects remain direct-module-only imports, are absent from every stable
package root and the production implementation-source inventory, and are not
constructed by the facade or consumed by a runtime.  The first real
`configuration -> systems` edge was authorized for the canonical Stokes
request.  At P2.1 completion the exact-edge ratchet permitted only
`plane_beris_edwards_components -> systems.stokes`; P2.3 later moves that same
edge with the physical component definitions to
`plane_beris_edwards_component_graph -> systems.stokes`.  `models`,
`planning`, and `runtime` receive no such permission.
P2.1 itself deliberately stopped before adding
`decompose_plane_beris_edwards_run_spec`; the separately qualified P2.2
follow-on described below now supplies that adapter without connecting a
production consumer.

The focused P2.1 component suite contains 98 passing tests.  The combined
P2.1/P2.1S, Stage F/G/H, import-boundary, and Phase 2 compatibility gate
contains 307 passing tests.  The complete local CPU suite contains 1419
passing tests and 8 passing subtests.  Fresh sdist and wheel archives plus an
isolated wheel installation pass direct aggregate import, provisional-export,
stable package-root non-export, and console-entry smokes.

### P2.1S: characterize and extract shared system declarations

The characterization portion of P2.1S was completed on 2026-09-19 without
changing production code.  Two machine-readable fixtures and three focused
test modules freeze:

- the algebraic and Stokes class, enum, field, signature, default, slots, and
  match-argument surfaces;
- execution-package and leaf-module object identity plus package-root
  non-exports;
- normalization, validation boundaries, metadata, typed/generic round-trip,
  and all three zero-mode policies;
- real legacy-module `AlgebraicUpdatePhase` protocol-4 class/member pickles
  with SHA-256 values
  `cdc6049ac6d32eaae433ad84520f99d4a3e256e6fa41ee99eb209434e29a1b9f`
  and
  `056186446f281a4fbda5bfe094d72941cc2794bed8b20613fdd18154ba7e53c8`;
- a real legacy-module Stokes protocol-4 pickle with SHA-256
  `f5e97c7453d9b2caf03c634c585797f368f0ac863193b509fb84f219b9d12488`;
- the existing negative contract that `AlgebraicSystemSpec` instances are not
  pickleable because their normalized parameters use `MappingProxyType`;
- representative Stokes metadata SHA-256
  `f89ce8db5693770e6c999b2171265e153c748d160093511f375c13c1eebb1c7e`;
- Plane algebraic restart-provenance SHA-256
  `e8808a592926bc47e855c7aaa246a0855606126b31f0a3850d6df6a3bdb24dbc`;
- fresh-process import ordering through the canonical and legacy paths.

The implementation portion of P2.1S was completed on 2026-09-19.
`AlgebraicUpdatePhase` and `AlgebraicSystemSpec` have one canonical definition
in `pssolver.systems.algebraic`.  `INCOMPRESSIBLE_STOKES_CAPABILITY`,
`PressureGauge`, `TangentialZeroModePolicy`, and
`IncompressibleStokesSystemSpec` have one canonical definition in
`pssolver.systems.stokes`.  The old `pssolver.execution.algebraic`,
`pssolver.execution.stokes`, and `pssolver.execution` paths are static
exact-object compatibility facades; no wrapper, duplicate, subclass, dynamic
import, runtime consumer, or numerical path was introduced.  The private
Stokes validation helpers live only in the canonical module.  The provisional
`pssolver.systems` package root intentionally exports no declarations.

The architecture ratchet recognizes `systems` as tensor-free and allows only
`systems -> {core, systems}`.  P2.1S introduced the first real
`execution -> systems` edge without pre-authorizing other consumers; the
subsequent disconnected P2.1 aggregate adds only the reviewed
`configuration -> systems` edge.  Models, planning, and runtime remain
unauthorized.  Existing production consumers continue importing through
`pssolver.execution`, which directly exercises the compatibility facades.
Stokes typed-to-generic conversion returns the exact canonical algebraic type.
The old Stokes pickle, representative metadata SHA-256, and Plane
restart-provenance SHA-256 remain unchanged.

The P2.1S focused suite now contains 91 passing tests.  The combined P2.1S,
Stage F/G/H, import-boundary, and Phase 2 compatibility/component gate
contains 273 passing tests.  The complete local CPU suite contains 1385
passing tests and 8 passing subtests.  Fresh sdist and wheel archives contain
the systems package, canonical algebraic and Stokes modules, and legacy
facades; an isolated wheel installation passes canonical and legacy identity,
old-pickle loading, new canonical pickle paths, unchanged negative
instance-pickle behavior, package-root non-export, Stokes lowering, and
console-entry smokes.  This mechanical migration changes declaration
ownership only: equations, runtime behavior, numerical operation order,
production provenance, and defaults are unchanged.  Its disconnected P2.1
follow-on is now complete; the next phase is the pure P2.2 facade
decomposition and parity adapter.

### P2.2: add pure decomposition and parity adapter

Implement decomposition from the facade to components.  Prove that every
legacy field can be recovered and that schema-v1 metadata, canonical hash,
runtime identity, and runtime-selection metadata are unchanged.  No runtime
consumer is migrated.  The round-trip includes raw preset inputs, dormant
requested friction, requested boundaries, and the separately recorded
qualified effective boundaries.

P2.2 was completed on 2026-09-19.  The leaf-only
`decompose_plane_beris_edwards_run_spec` adapter directly reads all 54 flat
facade fields and constructs the provisional component graph without calling
the facade's derived `geometry`, `numerics`, or `shendruk_preset` views.  A
test-only reverse projection recovers the 54 fields in their exact dataclass
order, reconstructs the facade, and then delegates every schema-v1 document
and identity calculation to the original facade authority.  There is no
second production serializer or reverse-composition API.

All seven schema-v1 golden cases, all 54 individually mutated fields, two CLI
paths, the format-v1 checkpoint identity, raw/resolved Shendruk inputs,
dormant/effective friction, requested/effective boundaries, and unnormalized
path spelling pass round-trip parity.  Existing weak direct-constructor
behavior is unchanged: malformed or noncanonical direct instances may still
be created exactly as before, while an explicit decomposition applies the
stronger component validation and can reject them.  No facade method, parser,
factory, production consumer, runtime, metadata authority, implementation
source inventory, default, or numerical path changed.

The P2.2-only suite contains 78 passing tests.  The focused component,
adapter, and import-boundary gate contains 183 passing tests.  The combined
P2.0/P2.1/P2.1S/P2.2, Stage F/G/H, and import-boundary gate contains 385
passing tests.  The complete local CPU suite contains 1497 passing tests and
8 passing subtests.  Fresh sdist and wheel archives, an isolated wheel
adapter/import smoke, stable-root non-export checks, and the installed console
entry smoke also pass.

At P2.2 completion the one-way dependency ran from
`plane_beris_edwards_components` to `plane_beris_edwards`, both for shared
Plane declarations and for the adapter's concrete
`PlaneBerisEdwardsRunSpec` input.  The supported facade was forbidden from
acquiring the reverse edge.  P2.3 therefore had to move shared declarations
to a dependency-neutral canonical leaf and split the facade-specific upper
adapter from a lower pure component graph before delegating facade views.
Bidirectional top-level imports, hidden dynamic imports, and local-import
workarounds remain forbidden.

### P2.3: prepare the dependency seam and delegate derived views

First establish the dependency-neutral declarations and split the upper
facade-specific decomposition adapter from the lower pure component builder.
Then let `domain`, `geometry`, `numerics`, and `shendruk_preset` delegate to
the component composition while preserving returned types and values.  Keep
all flat fields and compatibility properties.

The pre-extraction P2.3 characterization was completed on 2026-09-19 without
changing production code.  Its independent oracle freezes the nominal module
paths, dataclass and enum surfaces, exact facade/package object identities,
and real protocol-4 pickle bytes for the three shared Plane declarations,
representative values, and a complete RunSpec.  It also freezes all four
derived views across the seven schema-v1 golden cases, their uncached
freshness, and their field-local validation timing.  These gates must remain
green while the dependency-neutral declarations and pure builders are
introduced; the existing schema-v1, canonical-hash, runtime-identity, and
checkpoint oracles remain authoritative and unchanged.

The P2.3 implementation was completed on 2026-09-19.  The shared runtime-path,
free-slip-boundary, and spectral-refresh declarations now have one physical
definition in the dependency-neutral
`plane_beris_edwards_declarations` module.  Their supported nominal module,
old pickle globals and bytes, stable package exports, singleton identity, and
RunSpec defaults remain unchanged through exact-object facade aliases.  Four
narrow functions in `plane_beris_edwards_builders` now own construction of
the domain, geometry, numerics, and resolved Shendruk views.  The supported
facade calls only those narrow builders.

The six provisional component classes and explicit flat-value aggregate
builder are physically defined in
`plane_beris_edwards_component_graph`.  That lower graph imports declarations
and narrow builders but never the concrete facade.  The legacy
`plane_beris_edwards_components` path is a thin upper adapter: it imports the
concrete `PlaneBerisEdwardsRunSpec`, checks its type, forwards all 54 flat
values exactly once, and re-exports the exact component class objects under
their existing nominal path.  The facade imports neither graph nor adapter,
and all five modules are guarded by an exact acyclic-DAG test.  The graph may
be imported first without mutating the legacy string annotations; after the
legacy adapter path is loaded, `typing.get_type_hints` resolves exactly as it
did before extraction.  The lower aggregate builder remains
direct-module-only.

Only the declarations and narrow builders are recorded in production
implementation provenance because the supported facade executes them.  The
disconnected component graph and upper adapter remain absent from that
inventory and from every production runtime consumer.  The upper adapter
passes the original `pssolver.plane` Hermitian-axis policy explicitly; the
lower aggregate validates the same axis from the already-qualified Plane
topology instead of declaring a second numerical-policy authority.  P2.6 will
move that remaining production policy while removing the exact configuration
import debts.  No view constructs the full aggregate or caches values, and
validation timing is not broadened.

All 54 flat fields, 17 existing members, seven schema-v1 golden cases,
canonical and runtime hashes, format-v1 checkpoint identity, protocol-4
pickle bytes, and public exports remain unchanged.  The final P2.3-focused,
expanded, and complete CPU gates contain 161, 360, and 1581 passing tests,
respectively; the complete suite also contains 8 passing subtests.  Fresh
sdist and wheel archives contain declarations, narrow builders, the lower
component graph, and the upper adapter.  An isolated wheel installation passes
the six legacy component-class pickle hashes, exact declaration identities,
delegated view/component parity, direct-only export gates, and the installed
console-entry smoke.

### P2.4: freeze an explicit schema-v1 serializer adapter

Make the compatibility serializer an explicit authority without changing a
key, value, nesting level, list/tuple conversion, path spelling, JSON bytes,
or hash.  Component metadata remains internal.

P2.4 was completed on 2026-09-19.  Before extraction, a separate
characterization commit froze all seven schema-v1 documents and hashes,
fresh-document/mutation isolation, non-finite-value rejection, and the
historical dynamic dispatch between the six supported facade methods.  In
particular, canonical identity still dispatches through `self.to_metadata()`,
runtime identity still dispatches through `self.to_metadata()`, restart hash
still dispatches through `self.runtime_identity_metadata()`, and the compact
identity document still dispatches through `self.canonical_sha256()`.

The stateless direct-module-only
`PlaneBerisEdwardsSchemaV1CompatibilitySerializer` in
`plane_beris_edwards_schema_v1` is now the one implementation authority.  The
existing `PlaneBerisEdwardsRunSpec` methods remain the supported API and thinly
delegate to its singleton.  The schema-version constant is physically owned by
the serializer module and re-exported through the exact historical facade and
package paths.  No serializer type or singleton was added to a stable package
root.  The serializer has only a type-checking reference to the concrete
facade, has no dependency on the provisional component graph, and does not
normalize paths or reinterpret identities.  Component metadata remains
internal and is not a second production serializer.

Because this file now determines production configuration identity, it was
added to implementation-source provenance.  The four runtime/application
consumers remain untouched and continue to call the supported facade; their
one-at-a-time migration remains P2.5.  All 54 fields, all 17 members, the
factory and parser, seven golden metadata/JSON/hash cases, format-v1 checkpoint
identity, direct-constructor behavior, pickle identity, and production defaults
remain unchanged.

The final P2.4 serializer file contains 23 passing tests.  The expanded
compatibility, component, Fig. 4 CLI, and Stage O gate contains 369 passing
tests.  The complete local CPU suite contains 1604 passing tests and 8 passing
subtests.  Fresh sdist and wheel archives contain the serializer, an isolated
wheel installation reproduces facade/serializer metadata and hashes, and the
installed console-entry dry-run passes.  P2.4 changes no floating-point
operation, numerical runtime path, or GPU hot path, so no H100 job is required.

### P2.5: migrate consumers one per commit

Use this order:

1. `runtime/plane_legacy.py`;
2. `runtime/plane_beris_edwards.py`;
3. `workflows/plane_beris_edwards.py`;
4. `applications/plane_beris_edwards.py`.

The application comes last because it combines initialization, checkpoint
gating, runtime construction, workflow construction, and a second metadata
description.

#### P2.5.1: legacy runtime consumer complete

`pssolver/runtime/plane_legacy.py` is the first migrated consumer.  Commit
`140205cef8b01920a97becf5669f23a1326b84ef` froze its 27 direct flat-facade
reads, resolved object graph, short CPU trajectory, and exact restart behavior
before the production edit.  Commit
`50435e9299caa0059b164ecb1d46939c57d3e6a7` replaces those direct reads with
one call to
`decompose_plane_beris_edwards_run_spec` and consumes the resulting geometry,
numerics, physics, time-stepping, execution, workflow, and effective-boundary
components.

The migration preserves construction and floating-point operation order.  In
particular, it uses qualified effective boundaries rather than requested
boundary metadata and derives the existing Hermitian axis from the qualified
Plane geometry, including the full-complex rollback path.  It adds no stable
package-root export, changes no schema-v1 document or identity hash, and does
not change a runtime default.  The former
`runtime.plane_legacy -> pssolver.plane` import debt is retired.

The component adapter, lower component graph, active-nematic request
declarations, and Stokes system declarations now determine production runtime
meaning and are therefore included in implementation-source provenance.  This
is an implementation-identity change only.

Qualification against the pre-migration commit is byte-exact:

- continuous four-step `Q`, `u`, `p`, progress counters, and runtime identity;
- two-step state and the complete format-v1 checkpoint directory;
- candidate continuation from a baseline checkpoint;
- baseline continuation from a candidate checkpoint.

The focused gate contains 268 passing tests.  The complete CPU suite contains
1607 passing tests and 8 passing subtests.  Fresh sdist and wheel archives
contain every newly active module; an isolated installed-wheel one-step runtime
smoke and installed console-entry dry-run pass.  Because no floating-point
operation or GPU hot path changed, P2.5.1 requires no H100 job.  The next
consumer is `pssolver/runtime/plane_beris_edwards.py`.

#### P2.5.2: runtime selector consumer complete

`pssolver/runtime/plane_beris_edwards.py` is the second migrated consumer.
Commit `357a2e323f1cdf59a5a92ace97af0a20649b8062` freezes its two direct
flat-facade selector reads and the ordering of request validation, injected
legacy-builder validation, default legacy selection, and explicit lazy canary
import.  Commit `015b7f8d45cb80b804de2e1853e08383ecb4c30f` replaces the direct
`runtime_path` and `disable_q_gradient_reuse` reads with one selector-local
component decomposition and values from `components.execution`.

The supported flat facade remains the schema-v1 and mixed-authority identity
gate.  The default remains `legacy_production`, no fallback is introduced, and
the experimental canary module remains absent from the default process.  No
adapter, solver, timestep, kernel, boundary condition, numerical operation, or
implementation-source inventory is changed in this substep.

Both runtime paths were qualified against the pre-migration selector.  For
`legacy_production` and `separated_canary`, respectively, continuous four-step
and two-step segmented `Q/u/p`, initial-condition artifacts, complete format-v1
checkpoint directories, baseline-to-candidate restart, and
candidate-to-baseline restart are byte-identical.  Metadata is equal after the
expected output-path-derived canonical identity, elapsed-time, and
implementation-provenance allowances.

The focused gate contains 274 passing tests.  The complete CPU suite contains
1612 passing tests and 8 passing subtests.  Fresh sdist and wheel builds pass;
an isolated installed wheel completes one CPU step through each runtime path.
No floating-point operation or GPU hot path changed, so P2.5.2 requires no
H100 job.  The next consumer is
`pssolver/workflows/plane_beris_edwards.py`.

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
