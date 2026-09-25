# Phase 7 P7.7.9: qualified public compiler registry

Status: `P7_7_9_COMPLETE_P7_7_10_NOT_STARTED`.

Parent public-declaration baseline:
`0ba42120207c1ae7662a0e913ce4c1f6694fa116` on
`next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_9_TWO_COMBINATION_PUBLIC_COMPILER`.

P7.7.9 compiles the stable public `Simulation` declaration into one of the
two application combinations already qualified by the migration:

| equation variant | geometry | runtime paths |
|---|---|---|
| complete-stress Beris--Edwards | Plane slab | `legacy_production`, `compiled_v2` |
| legacy active-force active nematics | rectangular Channel | `legacy_channel`, `compiled_channel_v2` |

The machine-readable authority is
[phase_7_p779_public_runner_connection.json](phase_7_p779_public_runner_connection.json).

## Extensibility boundary

The two combinations are entries in an immutable qualified-compiler registry;
they are not encoded in the public `Simulation` dataclass. A new combination
is added by supplying:

1. an equation-system declaration and geometry declaration;
2. capability lowering and a geometry-solver requirement;
3. a runtime/package construction binding;
4. one application compiler adapter that preserves all declared identities;
5. numerical and performance qualification evidence.

Adding that entry does not change the public `Simulation` fields and does not
add model/geometry dispatch to the timestep. The public declaration layer can
represent an unregistered pair today. Compilation then returns the structured
rejection `unregistered_model_geometry`, including the requested capabilities
and registered pairs. This is a missing executable capability, not a claim
that models and geometries are permanently coupled.

Silently accepting every Cartesian product would be unsafe: different
topologies require different bases, Stokes solvers, nullspace policies, and
boundary semantics. Orthogonality means independently owned declarations and
extensible capability selection; it does not mean every mathematical product
already has a correct implementation.

## Compile-once contract

`compile_simulation(simulation)` performs all application selection before
allocation and returns an immutable `CompiledSimulation` containing:

- source declaration identity;
- normalized application `SimulationSpec`;
- capability lowering plan;
- package-owned construction plan;
- historical application RunSpec;
- explicit normalization evidence.

Both adapters forbid runtime fallback. The Plane adapter performs and audits
the existing fixed-K coefficient conversion. The Channel adapter is an exact
rho-parameterized translation and requires every initial-condition, PCG,
execution, workflow, boundary, and numerical field explicitly; it changes no
physical value.

## Scope correction

Commit `5f545bd` introduced the compiler together with a provisional Plane
`run_simulation()` connection before the planned phase boundary. This final
P7.7.9 completion keeps that backward-compatible Plane connection but does not
expand it. A compiled Channel request intentionally raises a clear P7.7.10
boundary error if passed to `run_simulation()`.

P7.7.10 still owns:

- dispatch to both qualified application runners;
- a common public result protocol;
- removal of application-specific return types from the public surface.

No rollback is needed because the provisional Plane call delegates once
before the timestep and changed no numerical path.

## Fail-closed behavior

Compilation rejects before allocation when:

- no registered adapter exists for the declared model/geometry pair;
- the requested runtime is not qualified for that pair;
- the integrator, boundary signature, topology, initial condition, numerical
  policy, pressure solver, execution policy, or workflow differs from the
  adapter contract;
- runtime fallback is requested;
- translation changes any declared scientific or operational identity.

The rejection is machine-readable. It does not choose a nearby geometry,
boundary law, model, solver, runtime, or default.

## Compatibility and performance

P7.7.9 adds no tensor operation, persistent allocation, numerical kernel,
transform, solver, checkpoint schema, output schema, or hot-path branch.
Selection and normalization occur once before construction. Existing Plane
and Channel production defaults remain unchanged, and Phase 8 is not
authorized by this work.

## Verification

Tests cover both runtimes for both registered application combinations,
stable construction identities, immutable compilation evidence, lossless
Channel translation, structured rejection of an unregistered cross-product,
and the explicit P7.7.10 runner boundary. The exact counts are recorded in the
machine-readable authority after the final targeted and complete CPU suites.

## Next boundary

The next task is P7.7.10: connect both compiled applications to
`run_simulation()` and introduce one stable public result protocol. P7.7.11
then owns CPU byte-identity and restart qualification, followed by the single
P7.7.12 H100 non-regression closure. Phase 8 remains deferred until those
steps pass.
