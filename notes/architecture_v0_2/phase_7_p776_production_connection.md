# Phase 7 P7.7.6: production application connection

Status: `P7_7_6_LOCAL_COMPLETE_H100_QUALIFICATION_PENDING`.

Parent baseline: `1078f44214d03fdce9326955f176e17aa36c0093` on
`next/pssolver-v0.2.0-architecture`.

Local classification:
`PASS_P7_7_6_LOCAL_PRODUCTION_CONNECTION_H100_PENDING`.

P7.7.6 connects both supported production applications to the package-owned
construction path introduced by P7.7.5.  It removes application-local runtime
builder selection without changing equations, numerical kernels, runtime
defaults, output arrays, or checkpoint state.

The machine-readable authority is
[phase_7_p776_production_connection.json](phase_7_p776_production_connection.json).

## Production call path

Both applications now follow the same construction sequence:

1. validate the existing run specification;
2. decompose it into the existing immutable component graph;
3. compose a tensor-free `SimulationSpec`;
4. lower and normalize it into a `PackageRuntimeConstructionPlan`;
5. pair that plan with the existing typed runtime request in a
   `PackageRuntimeConstructionInput`;
6. call `build_package_simulation_runtime()` exactly once;
7. pass the returned existing adapter directly to the existing workflow.

The Plane application no longer defines the compiled builder closure that
imported `build_plane_compiled_v2_runtime`.  Its qualified
`legacy_production` and `compiled_v2` paths use package construction.  The
older diagnostic-only `separated_canary`, which was deliberately outside the
P7.7.4/P7.7.5 binding set, retains one explicit call to its existing facade;
metadata labels it as a compatibility exception and package fallback remains
forbidden.  The Channel application no longer imports `build_active_nematic_channel`,
`build_channel_active_nematic_runtime`, or the compiled Channel bridge, and it
no longer defines legacy or compiled builder closures.  Those implementation
choices are package-owned behind the P7.7.5 factory.

## Identity and provenance

The factory still recomposes the simulation from the typed request and
requires exact equality with the supplied package plan before allocation.
For the four qualified paths, production metadata now records
`runtime_construction.plan` and `runtime_construction.input`, including the
simulation, lowering, binding, configuration, runtime-path, builder-ownership,
device, and no-fallback identities used by the connection.  The separated
canary instead records the exact compatibility owner and a null package plan.

This additive metadata is intentionally different from the parent baseline.
The scientific arrays and checkpoint schema are unchanged.

## Compatibility and performance boundary

The existing `legacy_production`, `compiled_v2`, `legacy_channel`, and
`compiled_channel_v2` adapters, solvers, transforms, state, workspace, and
`advance()` implementations are unchanged.  Planning and validation occur
once during construction.  There is no new dispatch, allocation, copy, or
identity calculation inside a timestep.

The implicit defaults remain:

- Plane: `legacy_production`;
- Channel: `legacy_channel`.

No package-root export is added.  The new construction types remain private
architecture interfaces, and selecting an unsupported or mismatched plan
still fails without fallback.

## Local qualification

Local tests cover both runtime paths for both geometries, verify that each
production application passes exactly one typed construction input, and bind
the emitted metadata to that exact input.  Existing Plane and Channel short
trajectory and same-runtime restart tests remain the numerical oracle.

Final local results:

```text
targeted: 181 passed
separated-canary compatibility recovery: 11 passed
complete: 2124 passed, 8 subtests passed
```

## H100 closure contract

The production entry points changed, so local success does not close P7.7.6.
One clean H100 qualification must compare this candidate with parent
`1078f44214d03fdce9326955f176e17aa36c0093` for all four runtime paths.
It must verify:

- the three known CUDA-only tests on the allocated H100;
- exact runtime selection and no fallback;
- exact construction-plan/request metadata;
- unchanged steady-state transform-call counts;
- baseline/candidate `Q/u/p` byte identity for matched short trajectories;
- same-runtime checkpoint/restart equivalence and existing cross-runtime
  rejection semantics;
- no NaN, Inf, OOM, CUDA error, graph break, or compile fallback;
- mean and median timestep non-regression within 3 percent and peak allocated
  and reserved memory non-regression within 5 percent.

No paired-win requirement is imposed for timing-equivalent paths.  Construction
wall time is reported separately from steady-state timestep time because this
slice deliberately moves one-time planning and validation, not numerical work.

Until that archive passes, `qualification_complete`, `p7_7_6_complete`, and
`phase_8_authorized` remain false.  This slice does not promote a compiled
runtime, alter a default, start Phase 8, or start Phase 9.
