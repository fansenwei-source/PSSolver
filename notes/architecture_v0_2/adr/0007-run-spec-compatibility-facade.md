# ADR 0007: decompose Plane run configuration behind a compatibility facade

Status: accepted for Phase 2.

## Context

`PlaneBerisEdwardsRunSpec` is a frozen, 54-field object that currently acts as
all of the following:

- the result of CLI and programmatic configuration resolution;
- a holder of scientific, discretization, execution, initialization, workflow,
  and provenance choices;
- the source of schema-v1 metadata and the full-run canonical hash;
- the source of the runtime/restart identity;
- the shared input to the application, runtime selector, legacy runtime, and
  workflow layers.

Replacing it directly with nested dataclasses would therefore change more than
code organization.  It could change its constructor, public attributes,
dataclass field order, equality and representation, metadata, hashes, dry-run
output, checkpoint acceptance, or the location and type of validation errors.

Several suitable value objects already exist.  In particular, `DomainSpec`,
`PlaneSlab`, `NumericsConfig`, `ShendrukPlanePreset`,
`SpectralRefreshSpec`, `IncompressibleStokesSystemSpec`,
`TangentialZeroModePolicy`, and `PressureGauge` must be reused rather than
recreated under Plane-specific names.  The executable
`BerisEdwardsPlaneCoupledModel` is not a suitable production declaration: it
depends on Torch and carries separated-canary execution and initialization
semantics.

## Decision

Phase 2 keeps `PlaneBerisEdwardsRunSpec` as the supported, flat compatibility
facade.  Its 54 fields, constructor, properties, factory signature, CLI,
schema-v1 serialization, canonical hash, runtime identity, and runtime
selection metadata remain compatible.

The new composition is provisional and internal.  It is introduced through a
pure decomposition function rather than by adding nested dataclass fields to
the facade.  The aggregate is composed from existing domain, geometry,
boundary, numerics, preset, refresh, and Stokes declarations plus the minimum
new Plane/model value objects described in the Phase 2 plan.

The compatibility direction is initially one-way:

```text
PlaneBerisEdwardsRunSpec (supported facade)
                    |
                    | pure decomposition
                    v
PlaneBerisEdwardsRunComponents (provisional internal aggregate)
```

Consumers migrate one at a time to component views.  The facade remains the
input and schema-v1 serializer until all consumers have moved and a separate
API decision authorizes a replacement.

The existing schema-v1 serializer remains the authority for:

- `to_metadata()`;
- `canonical_sha256()`;
- `runtime_identity_metadata()` and `runtime_identity_sha256()`;
- `identity_metadata()`;
- `runtime_selection_metadata()`.

Nested component metadata must not replace or augment schema-v1 output during
Phase 2.  In particular, historical placements such as `initial_s` under
`model` remain unchanged even though the new ownership model classifies it as
initial-condition state.

## Boundary compatibility decision

The `boundaries` field currently participates in metadata and runtime identity,
but the production application and legacy runtime construct their numerical
boundary constants from the global qualified free-slip preset.  An alternate
`PlaneFreeSlipBoundaryConditions` value can therefore change hashes without
changing the equations that run.

Phase 2 does not advertise this as boundary configurability.  The current
free-slip/free-Q declaration is the only qualified production value.  To
preserve direct-construction behavior, Phase 2 records both the requested
facade value and the effective qualified production value; it does not add a
new rejection or claim that an alternate request is executed.  Before a later
feature can treat boundaries as configurable, it must either introduce an
explicitly versioned fail-closed compatibility change or truly lower the
supplied physical assignment through the production runtime and qualify it
with manufactured and boundary-residual tests.

Silent metadata-only configurability is forbidden.  Strong anchoring and new
boundary mechanisms remain later features, not Phase 2 refactors.

## Identity decision

Phase 2 documents but does not redefine the existing identities.

- `canonical_sha256()` is the identity of the complete resolved run request.
  It includes output and workflow choices, initialization, runtime controls,
  and paths.  It is not a pure scientific identity.
- `runtime_identity_sha256()` is the same-backend restart identity.  It
  intentionally excludes output policy, most workflow fields, validation
  provenance, and every fresh-run initial-condition field, including
  `initial_s`.
- In `paper-window` parameterization, an unused raw `frank_k` does not affect
  resolved metadata or either hash.  `coefficient_max` affects validation but
  not the resolved metadata for an already-valid request.  In `fixed-k`, raw
  `frank_k` is effective, while `coefficient_min` and `coefficient_max` remain
  validation inputs rather than resolved identity values.
- Paths are serialized with `str(Path(...))`; Phase 2 does not resolve or
  normalize them.

The four identities described by ADR 0006 remain the architectural target,
but schema-v1 hashes are not silently reinterpreted as those new identities.
Any new identity schema requires a separate ADR and versioned migration.

## Module placement

Two new provisional implementation modules are planned initially, together
with one value object added beside the existing preset resolver:

- `pssolver/models/active_nematics/specifications.py` for geometry-neutral,
  tensor-free Beris--Edwards material parameters and the model-owned extruded
  defect-gas initial-condition request;
- the existing `pssolver/presets/shendruk.py` for a lossless raw
  `ShendrukPlaneParameterRequest` next to its derived `ShendrukPlanePreset`;
- `pssolver/configuration/plane_beris_edwards_components.py` for Plane time
  stepping, the Plane physics composition, execution, workflow/invocation,
  requested/effective compatibility values, and the aggregate composition.

These names are not added to a stable root public API during Phase 2.  Existing
generic types are imported from their current canonical modules.  No duplicate
`DomainSpec`, `NumericsConfig`, refresh policy, pressure gauge, or zero-mode
policy is introduced.

## Migration order

1. Add exhaustive compatibility characterization without production changes.
2. Add immutable component value objects without connecting them to a runtime.
3. Add pure facade decomposition and lossless round-trip/schema-v1 parity
   tests, including dormant requested friction and raw preset inputs.
4. Delegate existing derived properties to component views.
5. Migrate one consumer per commit: runtime construction, runtime selection,
   workflow, then application composition and metadata.
6. Remove the two exact configuration dependency debts on `plane` and
   `transforms` without replacing them with backend/operator dependencies.
7. Run the Phase 2 completion gates and retain the facade.

Moving all validation into `PlaneBerisEdwardsRunSpec.__post_init__`, changing
the direct-construction contract, changing identity semantics, or introducing
schema v2 is explicitly outside this sequence.

## Consequences

This approach is intentionally slower than replacing the RunSpec in one edit,
but it preserves the benchmark and restart oracle while ownership moves.  It
also prevents provisional Plane concepts from becoming premature general PDE
APIs.  The temporary cost is duplication between flat facade attributes and
derived component views; parity tests make that duplication explicit and
removable later.
