# Phase 7 P7.7.10: common runner and public result protocol

Status: `P7_7_10_COMPLETE_P7_7_11_NOT_STARTED`.

Parent compiler baseline:
`510941ae82a850bfd6afda436de09a235e30d355` on
`next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_10_COMMON_RUNNER_AND_RESULT_PROTOCOL`.

P7.7.10 connects the two application combinations qualified in P7.7.9 to
one public execution entry point:

```python
result = run_simulation(simulation)
```

The public runner accepts either a `Simulation` declaration or an already
compiled `CompiledSimulation`. A declaration is compiled exactly once before
runtime construction. The resulting application registration is dispatched
exactly once before the timestep; no model, geometry, boundary, or runtime
selection occurs inside a timestep.

The machine-readable authority is
[phase_7_p7710_public_runner_result.json](phase_7_p7710_public_runner_result.json).

## Qualified execution registry

The immutable application-runner registry now connects:

| application | qualified runtime paths | existing application runner |
|---|---|---|
| complete-stress Beris--Edwards on Plane | `legacy_production`, `compiled_v2` | `run_plane_beris_edwards` |
| legacy active-force active nematics in Channel | `legacy_channel`, `compiled_channel_v2` | `run_channel_active_nematics` |

Each registration is an application-level adapter. It resolves the existing
runner and output-directory semantics before execution, while the numerical
runtime and workflow remain the already qualified implementations. A future
model/geometry combination adds a compiler registration and a runner
registration; it does not add fields to the public `Simulation` declaration
or branches to the timestep.

An unqualified combination is still rejected by P7.7.9 as a structured
capability gap. P7.7.10 does not turn the current two-element registry into a
permanent whitelist and does not pretend that an unimplemented Cartesian
product already has a mathematically valid solver.

## Stable public result

`run_simulation()` now always returns the frozen `SimulationResult` contract.
It records:

- completion or dry-run status;
- selected application and runtime path;
- source declaration and application-request SHA-256 identities;
- resolved output directory;
- start/final steps and elapsed time;
- saved and checkpoint steps;
- final Q, velocity, and pressure observation;
- diagnostic records;
- lowering and construction provenance.

The public observation and diagnostic types are structural protocols. Plane
and Channel therefore expose the same stable fields without exposing either
`PlaneWorkflowResult` or `ChannelWorkflowResult` in the public API. The
wrapper reuses the observation arrays and diagnostic objects produced by the
application workflow; it does not copy the large Q/u/p arrays again.

`SimulationResult.to_metadata()` emits finite JSON-safe completion evidence
and array shape/dtype metadata without serializing array contents. Plane
dry-runs return an explicit `DRY_RUN` result with no false output directory,
steps, observations, diagnostics, or completion claim.

## Compatibility boundary

P7.7.10 changes only the staged public `run_simulation()` return contract and
adds Channel dispatch to that public entry. It does not change:

- either production application runner;
- tensor allocation, transforms, Stokes solves, or timestep kernels;
- numerical coefficients, boundary conditions, or runtime defaults;
- checkpoint, observation, diagnostic, metadata, or on-disk output schemas;
- Plane or Channel command-line production entry points.

Application modules remain lazily imported, so importing `pssolver` does not
eagerly import Plane/Channel runtime, workflow, or tensor execution modules.
The public wrapper adds constant-size Python metadata only after a workflow
finishes, so it adds no persistent GPU allocation and no per-step overhead.

## Failure semantics

The public runner fails closed when:

- a compiled application has no registered execution adapter;
- an application unexpectedly returns no result outside a qualified Plane
  dry-run;
- an application result lacks the common workflow fields;
- completion steps, elapsed time, observation identity, or diagnostics do not
  satisfy the public protocol;
- provenance cannot be represented as finite canonical JSON.

There is no runtime fallback to another application, model, geometry,
boundary condition, solver, or runtime path.

## Verification boundary

CPU tests cover both Channel runtime paths, Plane dispatch and dry-run
semantics, compile-before-dispatch behavior, one-call application dispatch,
public result immutability, array identity preservation, lazy imports,
JSON-safe metadata, root exports, and historical P7.7.9 evidence stability.

P7.7.10 does not claim new numerical qualification. P7.7.11 must compare the
new public entry against the existing Plane and Channel entries for CPU
byte-identity and restart behavior. P7.7.12 owns the single H100
non-regression closure. Phase 8 and Phase 9 remain unauthorized.

## Next boundary

Proceed to P7.7.11 without changing the numerical path: freeze small Plane
and Channel CPU inputs, compare old and public entries byte-for-byte, verify
same-runtime restart behavior, and confirm that public result metadata agrees
with the existing workflow evidence.
