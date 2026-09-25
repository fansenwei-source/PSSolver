# Phase 7 final closure

Status: `PASS_PHASE7_CHANNEL_AND_PUBLIC_SIMULATION_ARCHITECTURE_CLOSURE`.

Phase 7 is complete. Its first track migrated and independently qualified the
rectangular Channel as the second geometry. Its P7.7 track then introduced the
tensor-free `Simulation` declaration, explicit model--geometry--boundary
composition, capability lowering, package-owned runtime construction, and the
common `compile_simulation()` / `run_simulation()` / `SimulationResult` public
protocol.

## Numerical closure

The Channel P7.6 closure established byte identity between `legacy_channel`
and `compiled_channel_v2` for a matched 1000-step H100 run. The production
connection P7.7.6 closure qualified the package-owned construction route,
restart behavior, negative identity gates, performance, and memory. P7.7.12
then established direct/public scientific identity and common-final-step
restart diagnostics for the public API.

Plane and Channel remain independently qualified; a Plane result is not used
as evidence for a Channel-specific Stokes, pressure, wall, or warm-start
property. No runtime fallback is allowed.

## Architecture at closure

Public simulations are tensor-free declarations. Model, geometry, boundary,
numerical, time, initial-condition, execution, and output ownership are
separate declaration fields. Compilation resolves a registered application
before allocation. Application dispatch occurs before the timestep and never
inside it. Runtime construction is package-owned, and completion is returned
through an application-neutral result protocol.

Two application combinations are currently executable:

1. complete-stress Beris--Edwards on a Plane slab, through
   `legacy_production` or `compiled_v2`;
2. the existing active-force Channel model on a rectangular Channel, through
   `legacy_channel` or `compiled_channel_v2`.

An unregistered combination can be declared but fails compilation with a
structured capability gap. This is an extensibility boundary, not a claim that
arbitrary combinations are already implemented or qualified.

## Compatibility and authorization

`legacy_production` and `legacy_channel` remain the defaults. No compiled
runtime is promoted. The closing records change no numerical kernel,
checkpoint schema, model equation, boundary physics, or output schema.

Phase 8 planning is now eligible. Phase 8 execution is not authorized by this
closure. Strong anchoring, nonhomogeneous lifting, additional boundary laws,
new geometry/model combinations, and arbitrary capability registration remain
future work. Phase 9 and optimal control also remain unauthorized and outside
this repository's current closure.

The local closure audit passed 15 targeted tests and the complete suite of
2199 tests plus 8 subtests. The archive builder now lists the full Phase 7
evidence chain, but the existing verbatim PDF was deliberately not regenerated.

The machine-readable authority is
[phase_7_final_closure.json](phase_7_final_closure.json).
