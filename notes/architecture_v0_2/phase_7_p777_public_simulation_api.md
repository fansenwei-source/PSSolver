# Phase 7 P7.7.7: public `Simulation` API contract

Status: `P7_7_7_COMPLETE_P7_7_8_NOT_AUTHORIZED`.

Parent baseline: `a739fbc0cb4c7addce5fab0cf259a65fa7627ad6` on
`next/pssolver-v0.2.0-architecture`.

Classification: `PASS_P7_7_7_PUBLIC_SIMULATION_API_CONTRACT`.

P7.7.7 freezes the first supported package-root declaration for an orthogonal
simulation request. It adds no convenience model, geometry, boundary,
numerics, time, execution, or output builders; it does not connect
`run_simulation`; and it cannot allocate a tensor or enter a timestep.

The machine-readable authority is
[phase_7_p777_public_simulation_api.json](phase_7_p777_public_simulation_api.json).

## Public contract

The supported import introduced by this slice is:

```python
from pssolver import Simulation
```

`Simulation` is immutable and owns the following constructor fields in this
exact order:

```text
model
geometry
boundaries
numerics
time
initial_condition
execution
output
discretization
invocation
```

The fields accept the canonical tensor-free declarations already used by the
P7.7 composition and lowering path. The resulting object exposes its validated
`SimulationSpec` through `simulation.specification` and delegates canonical
metadata, the complete declaration SHA-256, and scientific, discretization,
execution, and run identities to that specification.

P7.7.7 intentionally does not export `run_simulation`. A declaration is not
an executable runtime, and the later connection must not silently choose a
nearby qualified application or fall back between runtime paths.

## Field ownership

- `model` owns boundary-free equations, physical parameters, algebraic
  subsystem requests, diagnostics, and admitted initial-condition families.
- `geometry` owns the domain grid, physical lengths, axes, and topology. It
  owns no physical wall law.
- `boundaries` attaches physical or algebraic-compatibility laws to scalar
  components and oriented faces. It owns no DCT/DST implementation choice.
- `numerics` owns precision, dealiasing, transform intent, and spectral
  storage.
- `time` owns the mathematical integrator, timestep, startup, and refresh
  identity. It does not own total duration.
- `initial_condition` owns the realization family, source, parameters, and
  seed.
- `execution` owns backend, runtime path, device, compilation, cache, and
  implementation policy.
- `output` owns finite duration, observation and save schedules, checkpoint,
  restart, and output-location policy. The field name is user-facing; its
  canonical type remains the existing tensor-free `WorkflowSpec`.
- `discretization` carries algebraic-solver and other discretization controls
  not represented by the generic numerical declarations. Its mapping is
  normalized to finite canonical JSON by `SimulationSpec`.
- `invocation` owns non-scientific command and provenance inputs.

This separation preserves the four identities frozen by ADR 0006. In
particular, total steps remain run identity rather than time-discretization
identity, and device changes remain execution identity rather than scientific
identity.

## Construction and validation

Construction creates one canonical, tensor-free `SimulationSpec`. Existing
cross-object checks therefore remain authoritative:

1. model, geometry, boundary, numerical, time, initial-condition, execution,
   output, and invocation declarations have their exact canonical types;
2. geometry and boundary dimensions agree;
3. boundary assignments cover exactly evolved and algebraic components;
4. physical and algebraic boundary semantics remain distinct;
5. periodic and bounded topology agrees with every face law;
6. generated initial-condition families are declared by the model;
7. additional discretization controls are finite and JSON-compatible.

No compatibility metadata is accepted through the public constructor. Legacy
adapter evidence remains owned by compatibility composition, not by a new
scientific request.

## Performance and dependency boundary

`pssolver.api.simulation` imports no Torch, NumPy, backend, transform,
operator, linear solver, runtime, workflow implementation, or application.
Its P7.7.7 source test independently freezes that tensor-free dependency
boundary without rewriting the historical Phase 0 ratchet file whose source
identity is part of earlier immutable qualification evidence.

P7.7.7 changes no equation, boundary law, transform, solver, runtime,
workflow, output format, checkpoint schema, production default, or numerical
kernel. It adds no hot-path dispatch and no persistent device allocation.

## Compatibility and authorization boundary

All existing application entry points remain supported and unchanged:

- `run_plane_beris_edwards`;
- `run_channel_active_nematics`;
- the Plane CLI and its compatibility module.

P7.7.7 does not introduce the desired convenience imports such as
`CompleteStressBerisEdwards`, `free_slip_velocity`, `neumann_q`, typed public
numerical/output objects, or ergonomic `PlaneSlab(shape=..., lengths=...)`.
Those belong to a separately authorized P7.7.8.

It also does not connect `run_simulation`, add a new executable model/geometry
combination, add boundary physics, promote a compiled runtime, authorize
Phase 8, or authorize Phase 9.

## Verification

The P7.7.0--P7.7.7 architecture/API target set passed:

```text
117 passed
```

The final public-root and transform-compatibility target set passed:

```text
30 passed
```

The complete local CPU suite passed:

```text
2137 passed, 8 subtests passed
```

No test failed, skipped, or was deselected. H100 qualification is not required
because this slice cannot construct a runtime, execute a timestep, or allocate
device state.

## Next slice

P7.7.8 is not authorized by this result. If separately authorized, it may add
typed public convenience objects that normalize into the declarations frozen
here. It must preserve this constructor shape, keep dictionaries and string
dispatch outside the timestep, and leave `run_simulation` disconnected until
an independent compiler/runner slice is qualified.
