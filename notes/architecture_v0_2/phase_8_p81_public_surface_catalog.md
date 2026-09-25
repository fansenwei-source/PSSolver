# Phase 8 P8.1: existing-capability public surface and catalog

Status: `PASS_P8_1_EXISTING_CAPABILITY_PUBLIC_SURFACE_AND_CATALOG`.

Baseline: `f8789461ee71391885d5e04c6906a13bc83a26b0` on
`next/pssolver-v0.2.0-architecture`.

P8.1 makes capabilities that were already scientifically qualified usable
through the public declaration layer. It adds no equation variant, compiler
registration, boundary physics, transform, runtime, timestep, checkpoint
schema, or production-default change. Its machine-readable closure record is
[phase_8_p81_public_surface_catalog.json](phase_8_p81_public_surface_catalog.json).

## Public model facade

`LegacyActiveForceActiveNematics(...)` now constructs the canonical
`legacy_active_force_active_nematics` equation declaration used by the
qualified Channel application. All seven coefficients are required:

```python
from pssolver.models.active_nematics import (
    LegacyActiveForceActiveNematics,
)

model = LegacyActiveForceActiveNematics(
    rho=6.0,
    elastic_constant=1.0,
    activity=5.0,
    beta=-1.0,
    flow_alignment=1.0,
    friction=0.0,
    viscosity=1.0,
)
```

The facade selects no geometry, wall law, pressure algorithm, numerical
method, initial condition, runtime, or output policy. For matching
coefficients its returned `EquationSystemSpec` is equal to the existing
Channel application declaration.

## Public geometry facades

`PeriodicBox` and `RectangularChannel` now accept the same ergonomic
`shape`/`lengths` form as `PlaneSlab`, while retaining the original
`DomainSpec` form:

```python
from pssolver.geometries import PeriodicBox, RectangularChannel

box = PeriodicBox(
    shape=(128, 128, 64),
    lengths=(40.0, 40.0, 20.0),
)
channel = RectangularChannel(
    shape=(512, 40, 40),
    lengths=(128.0, 10.0, 10.0),
    streamwise_axis=0,
)
```

These are compatibility constructors, not new geometry implementations. Each
returns the historical canonical object from
`pssolver.geometries.tensor_product`; concrete type, equality, hashing,
pickle identity, registry identity, and restart provenance are unchanged.

## Public no-slip policy

`no_slip_velocity()` declares homogeneous Dirichlet conditions for every
velocity component on every bounded face. Periodic axes remain periodic. The
policy expands through the same model-neutral `assign_boundaries(...)` helper
as the existing Q, free-slip velocity, and pressure-compatibility policies.
Its component-level identity exactly matches the existing qualified Channel
wall law.

Boundary-assignment labels are descriptive provenance, not physical identity.
The Channel public compiler now compares dimension and component-face laws,
not the arbitrary assignment name. This allows a public assignment built by
`assign_boundaries(...)` to compile without knowing an internal legacy label.

## Immutable capability discovery

The root package and `pssolver.api` expose:

```python
from pssolver import (
    available_models,
    available_geometries,
    available_boundary_policies,
    available_combinations,
    capability_catalog,
)
```

The first three functions list declarable constructors. Their immutable
records identify which already-qualified applications, if any, use each
declaration. `available_combinations()` is derived from the live compiler
registry and lists only executable model--geometry pairs and their qualified
runtime paths. `capability_catalog()` groups both views in one frozen value
object with a JSON-safe `to_metadata()` representation.

This distinction is intentional. `PeriodicBox` is public and declarable, but
P8.1 reports it as not yet executable because no periodic compiler adapter is
registered. Constructing it does not imply a solver exists. An attempted
unregistered combination continues to fail before allocation with
`UNREGISTERED_MODEL_GEOMETRY`.

## Qualified matrix after P8.1

The executable registry remains exactly:

| Equation variant | Geometry | Runtime paths |
|---|---|---|
| `complete_stress_beris_edwards` | `plane_slab` | `legacy_production`, `compiled_v2` |
| `legacy_active_force_active_nematics` | `rectangular_channel` | `legacy_channel`, `compiled_channel_v2` |

No periodic pair, cross-geometry pair, new anchoring law, or generic fallback
was registered.

## Compatibility maintenance

The P7.7.8 source hashes are historical phase evidence. Its test now checks
that the archived record retains the complete reviewed-source set and valid
SHA-256 identities rather than incorrectly requiring later Phase 8 source
files to keep their P7 bytes. P8.1 records the current source identities in
its own closure record.

## Authorization boundary

P8.1 is locally complete and makes P8.2 planning eligible. It does not
authorize P8.2 implementation, a periodic Stokes runtime, an H100 job, Phase
9, or production-default promotion. P8.2 must separately manufacture-test,
lower, register, and qualify complete-stress Beris--Edwards on `PeriodicBox`.
