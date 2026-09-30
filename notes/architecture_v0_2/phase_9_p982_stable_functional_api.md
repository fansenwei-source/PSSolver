# Phase 9 P9.8.2: stable functional protocol and compatibility facade

## Result

P9.8.2 publishes functional protocol version `1.0` from the installed-package
module `pssolver.functional.api`.  The broader `pssolver.functional` module
remains the compatibility-public facade for the already-qualified P9.6 and
P9.7.6 consumers.  This split prevents qualification-only helpers and concrete
runtime implementation classes from becoming stable merely because they were
historically present in `__all__`.

The machine-readable authority is
[phase_9_p982_stable_functional_api.json](phase_9_p982_stable_functional_api.json).

## Stable surface

The stable module exports 46 exact names.  They cover:

- tensor, state, control, observation, identity, request, declaration,
  runtime, factory, and checkpoint contracts;
- the periodic and Channel batch-one request/build entry points already
  qualified by independent consumers;
- protocol-version negotiation and package/protocol provenance;
- a structured functional exception hierarchy;
- checkpoint compatibility metadata;
- the stable Channel pressure-convergence diagnostic schema.

`FunctionalCapabilities` is deliberately absent from the stable module.  It
remains a compatibility-public alias of `FunctionalCapabilitySet`, is recorded
as deprecated in package version 0.2.0, and cannot be removed before 0.4.0.
This supplies the two-minor-release window required by ADR 0005 without
breaking an existing import.

## Version negotiation

New construction accepts exactly protocol `1.0`.  Checkpoint reading accepts
exactly `1.0` and the qualified legacy value `0.1-provisional`.  The legacy
value is read-only: it cannot construct a new runtime and cannot be silently
selected as a fallback.

Negotiation returns a typed, JSON-safe record containing requested version,
effective version, purpose, legacy status, and the exact compatibility reader.
Unknown versions fail with `FunctionalVersionError`; no version prefix,
semantic-version range, or heuristic repair is used.

Machine-readable provenance also binds PSSolver package version 0.1.2,
protocol version 1.0, supported construction and checkpoint-read versions,
compatibility-policy version 1, and the alias deprecation window.

## Checkpoint compatibility

The qualified periodic and Channel version-1 schemas remain unchanged on
disk.  New exports write protocol 1.0 and its corresponding functional runtime
identity.  Imports additionally recognize the exact legacy
`0.1-provisional` bridge metadata and recompute the exact legacy identity
digest from the current target identity with only the API-version literal
changed.

Every successful import now returns `FunctionalCheckpointCompatibility` with:

- source and target API version;
- the exact reader name;
- whether the checkpoint used the current protocol exactly.

Production checkpoints without embedded functional protocol metadata remain
supported through their existing qualified production readers and are marked
as production compatibility paths.  The two version-1 formats remain distinct;
no Periodic/Channel interchange, missing-field synthesis, or checksum repair is
allowed.

## Structured refusal semantics

`pssolver.functional.api` normalizes public-boundary failures into stable
exceptions with a machine-readable `code`, operation, JSON-safe details, and
`to_metadata()`.  Type and value errors remain subclasses of Python
`TypeError` and `ValueError`; checkpoint path errors remain subclasses of
`FileNotFoundError` and `FileExistsError`.  Existing callers can therefore
retain their current catches while new consumers avoid parsing message text.

The stable hierarchy distinguishes contract, version, identity, checkpoint,
checkpoint integrity, checkpoint compatibility, execution, and convergence
failures.  The compatibility facade retains the already-qualified behavior.

## Pressure-convergence meaning

`ChannelPressureSolveDiagnostics` now reports the stable schema required by
the P9.7.6 review:

- solver and operator identity;
- convergence mode;
- requested iteration limit and achieved iteration count;
- achieved absolute and relative residual;
- requested relative tolerance;
- tolerance semantics;
- termination reason;
- an explicit acceptability decision.

Fixed-iteration configuration is named
`deterministic_iteration_cap_with_early_convergence`.  Its tolerance is an
early-convergence criterion, not a required postcondition after the cap.  The
non-fixed configuration retains a required relative-residual postcondition
with a maximum-iteration cap.  The old `iterations`, `residual`, and
`relative_residual` attributes remain read-only compatibility properties.

This is metadata stabilization only.  PCG arithmetic, initial guesses,
operators, iteration order, stopping branches, warm-start behavior, spectral
discretization, and timestep execution are unchanged.

## Authorization boundary

P9.8.2 makes P9.8.3 planning eligible.  It does not migrate either independent
consumer, complete installed-wheel packaging qualification, authorize H100,
change a production default, qualify a larger batch, authorize an optimizer
campaign, or complete Phase 9.  Regenerating the cumulative PDF remains a
separate explicit user action.
