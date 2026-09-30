# Phase 9 P9.8.1: public functional API and schema inventory

## Result

P9.8.1 is complete as a read-only inventory.  It changes no import, runtime,
checkpoint, numerical, or production path.  The machine-readable authority is
[phase_9_p981_public_api_surface_inventory.json](phase_9_p981_public_api_surface_inventory.json).

The audited provider baseline is
`1fade33cf42a9f4bca6745fa5848c93fb3662053`.  The functional interface remains
`0.1-provisional` and remains available only from `pssolver.functional`; it is
not re-exported from the package root.

## What the two consumers actually use

`pssolver.functional.__all__` contains 57 names.  The two qualified independent
consumers directly import a union of only 12:

- `FUNCTIONAL_API_VERSION`;
- `CHANNEL_ACTIVITY_RUNTIME_STAGE`;
- `FunctionalCapabilitySet`;
- `FunctionalControlFieldSpec`;
- `FunctionalRuntimeIdentity`;
- `FunctionalRuntimeProtocol`;
- `FunctionalStateSpec`;
- `FunctionalTensorSpec`;
- `build_functional_runtime`;
- `periodic_activity_functional_request`;
- `build_channel_activity_functional_runtime`;
- `channel_activity_functional_request`.

They additionally depend behaviorally on the public runtime protocol
properties and methods, and the Channel consumer uses the checkpoint bridge
returned by the runtime.  Those indirect protocol dependencies are part of
the stabilization inventory even when the consumer does not import their type
names directly.

## Export classification

The 57 exports are partitioned into four review groups in the JSON record:

1. **core contract candidates** -- tensor/state/control/observation specs,
   runtime identity, protocols, and type aliases;
2. **qualified runtime entry candidates** -- the two qualified builders,
   request functions, runtime implementations, factories, bridges, and their
   version constants;
3. **compatibility alias review** -- `FunctionalCapabilities`, which is an
   alias of `FunctionalCapabilitySet` and must not silently become a second
   semantic type;
4. **provider primitive and qualification-only review** -- pressure
   transpose/implicit-adjoint primitives, the small-grid unrolled oracle, and
   frozen validation helpers and reports.

This is an inventory, not a stability decision.  Being in `__all__` or having
been used by a qualification harness is not sufficient for stable-public
status under ADR 0005.

## Runtime protocol

Both runtime implementations satisfy the same public shape:

- properties: `api_version`, `state_spec`, `control_specs`,
  `control_field_schema`, `observation_specs`, `capabilities`, and
  `checkpoint_bridge`;
- identity/construction: `identity()` and `initial_state()`;
- execution: `step(state, controls, step_index)`,
  `observe(state, controls_or_none)`, and
  `step_and_observe(state, controls, step_index)`.

The factory protocol owns `build(request)`.  The checkpoint bridge protocol
owns `format_version`, `export_checkpoint(...)`, and `import_checkpoint(...)`.
All input tensor shape, dtype, device, batch-axis, order, and admissibility
checks are fail-closed; no implicit conversion is part of the contract.

## Identity and metadata schemas

The canonical runtime identity contains `api_version`, `scientific`,
`discretization`, `execution`, and `state_layout`.  It is serialized as JSON
with sorted keys, compact separators, and NaN/Inf rejection, then hashed with
SHA-256.  Python object identity and memory addresses are excluded.

The inventory freezes the current metadata key sets for tensor, state,
control, observation, capability, construction-request, declaration, and
checkpoint-state contracts.  P9.8.2 may add explicit schema/version wrappers,
but it may not silently reinterpret an existing key or hash input.

## Checkpoint schemas

The periodic bridge format version is 1.  Its functional bridge metadata is
embedded in the production workflow checkpoint and binds API version, runtime
kind, functional and production identities, state layout, and the layout
digest.

The Channel functional bridge format version is also 1, but it owns a
functional-only `checkpoint.json` plus two `.npy` tensors.  Its top-level
metadata contains format kind/version, completed steps, bridge identity, and
tensor records.  Each tensor record binds filename, shape, dtype, and SHA-256.
The bridge may also import Q from a production Channel checkpoint after
validating runtime, backend, and tensor identities.  Pressure and pressure
warm-start state are deliberately absent from the functional checkpoint.

The two version numbers both equal 1 but name different formats; they are not
interchangeable.

## Error taxonomy gap

The provisional provider currently uses built-in `TypeError` for type/dtype
violations, `ValueError` for shape/device/identity/capability/nonfinite
violations, `FileNotFoundError` and `FileExistsError` for checkpoint paths,
and `FunctionalValidationError` for frozen qualification failures.  There is
no common stable functional-contract exception hierarchy yet.  Consumers map
provider-contract refusal into their own integration exception.

P9.8.2 must define a stable, machine-distinguishable error policy without
weakening current fail-closed checks or forcing consumers to parse error
message text.  Compatibility with the current built-in exception behavior
must be explicit.

## Authorization boundary

P9.8.1 completes the surface and schema inventory and makes P9.8.2 planning
eligible.  It does not select or publish a stable version, change any export,
add an exception class, migrate a consumer, authorize H100 work, change a
production default, qualify larger batches, or complete Phase 9.
