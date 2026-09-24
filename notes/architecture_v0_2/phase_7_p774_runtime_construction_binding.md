# Phase 7 P7.7.4: opt-in runtime construction binding

Status: `P7_7_4_COMPLETE_P7_7_5_NOT_AUTHORIZED`.

Parent baseline: `2393362ae9816b9329d8e72f87acc1eeeb137743` on
`next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_4_OPT_IN_RUNTIME_CONSTRUCTION_BINDING`.

P7.7.4 creates the first executable, but still opt-in, bridge from the generic
`SimulationSpec` and `SimulationLoweringPlan` to the already-qualified Plane
and Channel runtime factories. It does not replace either factory, change an
application entry point, or move dispatch into a timestep.

The machine-readable authority is
[phase_7_p774_runtime_construction_binding.json](phase_7_p774_runtime_construction_binding.json).

## Two-stage binding

Construction is split deliberately:

1. `bind_simulation_runtime()` is tensor-free. It verifies or constructs the
   exact lowering plan and returns a frozen `RuntimeConstructionBinding`.
2. `build_bound_simulation_runtime()` recomposes the declaration from the
   existing typed build request, requires exact binding equality, validates
   builder ownership, and delegates to the existing geometry runtime factory.

The binding records both the source simulation SHA-256 and lowering-plan
SHA-256, together with equation variant, geometry, runtime path, request type,
factory, adapter protocol, concrete geometry solver, and builder ownership.
It stores names, not imported callables or mutable registries.

This ensures that a binding produced for one grid, boundary contract, runtime
path, solver option set, or initial-condition declaration cannot be reused for
a different request.

## Closed target set

The binding whitelist contains exactly four previously qualified targets:

- complete-stress Beris--Edwards / Plane / `legacy_production`;
- complete-stress Beris--Edwards / Plane / `compiled_v2`;
- legacy active-force active nematics / Channel / `legacy_channel`;
- legacy active-force active nematics / Channel / `compiled_channel_v2`.

`separated_canary` is intentionally not connected by this generic path. No
nearest match or runtime fallback is permitted. A supported model/geometry
with an unqualified runtime path raises a structured
`SimulationBindingError` before runtime construction.

## Builder ownership is explicit

P7.7.4 records the architecture as it actually exists rather than pretending
that construction has already been normalized:

- Plane legacy construction is package-owned;
- Plane compiled construction still requires `compiled_builder` from the
  caller;
- Channel legacy construction still requires `legacy_builder` from the
  caller;
- Channel compiled construction still requires `compiled_builder` from the
  caller.

Only the selected builder parameter may be supplied. Missing caller-owned
builders and extra unselected builders fail before delegation. This makes the
remaining application/runtime coupling measurable and gives P7.7.5 a narrow
target without changing numerical code prematurely.

## Zero-copy delegation and hot-path cost

The connector returns the exact object returned by the existing runtime
factory. It adds no adapter wrapper, tensor copy, registry, or field-name
lookup. Binding validation happens once before construction. Existing solver,
integrator, transform, model, workspace, and timestep objects are untouched.

Consequently, an already-built runtime executes the same `advance()` method
and the same numerical kernels as before. The connector has no timestep-time
or persistent GPU-memory cost.

## Equivalence evidence

Tests establish two complementary contracts:

- a factory sentinel passes through the connector with Python object identity
  unchanged;
- independent Plane legacy and Channel legacy two-step CPU runs constructed
  through the old factory and the new bound path have byte-for-byte identical
  physical and spectral fields, linear state where applicable, and progress.

The four binding identities are tested independently. Existing compiled
runtime factory and long-run equivalence evidence remains authoritative; this
slice does not duplicate or reinterpret those qualification results.

## Public and production boundaries

The new modules are not exported from `pssolver`, `configuration`, `planning`,
or `runtime` package roots. Neither production application imports the new
connector. Plane still defaults to `legacy_production`; Channel still defaults
to `legacy_channel`.

No checkpoint, output, metadata, CLI, pickle, or public API contract changes.
No production default is promoted.

## Verification policy

P7.7.4 is locally testable because the new execution edge is a validating,
direct delegation and both geometry paths have byte-exact CPU trajectory
tests. The final results are:

```text
targeted: 190 passed
complete: 2104 passed, 8 subtests passed
```

An H100 run is not required to accept this disconnected opt-in slice. H100
qualification remains mandatory before a later change connects production
applications or changes a default, because that would expose the binding edge
to production performance and memory behavior.

## Next slice

P7.7.5 is not authorized by this result. A later slice may normalize the
remaining application-owned builder assembly into typed package construction
inputs. It must continue to reuse the qualified numerical kernels, preserve
direct adapter identity, leave production applications opt-in until separate
qualification, and avoid construction dispatch inside timesteps.

P7.7.4 does not add boundary physics, make Plane and Channel use one solver,
promote a compiled runtime, authorize Phase 8, or authorize Phase 9.
