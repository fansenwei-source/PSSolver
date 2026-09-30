# Phase 9 P9.8.3: stable functional API consumer migration

Date: 2026-09-30

Status: complete

P9.8.3 migrates both independent PSSolver-Control consumers from the broad
`pssolver.functional` compatibility facade to the stable installed-package
boundary `pssolver.functional.api`, protocol version `1.0`.  The machine-
readable record is
[phase_9_p983_consumer_migration.json](phase_9_p983_consumer_migration.json).

## Bound implementations

Provider:

- repository: `fansenwei-source/PSSolver`;
- branch: `next/pssolver-v0.2.0-architecture`;
- stable API baseline: `1ebafc22c6ed75ea5ba70a508724eddd8a090ef2`;
- stable module: `pssolver.functional.api`;
- construction protocol: `1.0`.

Consumer:

- repository: `fansenwei-source/PSSolver-Control`;
- branch: `integration/p9.7-channel-functional-consumer`;
- migration implementation: `b830865b1fbea07ec708a3af181a136f84559977`;
- migration evidence record: `ea24027`;
- periodic and Channel adapter identity versions: 2.

The consumer implementation commit contains the code, live provenance,
documentation, and tests.  The following commit adds the machine-readable
P9.8.3 record without changing the adapters.  This two-commit structure avoids
placing an unknown or self-referential commit identifier in qualification
metadata.

## Dependency boundary

Both live adapters import only `pssolver.functional.api`.  They pin exact
construction version `1.0` and fail closed on a different module/runtime
version.  They do not import the compatibility facade, private runtime
modules, `pssolver.control`, solver storage, fields, transforms, or integrator
internals.  Dependency direction remains:

```text
PSSolver-Control -> PSSolver public API
PSSolver -/-> PSSolver-Control
```

The migration changes neither the model equations nor the control algorithms.
It changes the public module path, protocol identity, adapter identity, and
run provenance only.

## Historical evidence and compatibility

The P9.6 periodic and P9.7 Channel candidate/H100 records remain byte-for-byte
unchanged.  They continue to bind the provisional `0.1-provisional` protocol,
their original provider commits, and their original consumer sources.  They
are evidence that the two real consumers existed and were independently
qualified before stabilization; they are not relabelled as stable-protocol
installed-wheel evidence.

New functional runtime construction accepts only protocol `1.0`.  Existing
qualified periodic and Channel checkpoints with declared version
`0.1-provisional` remain readable through PSSolver's exact, provider-owned
compatibility readers.  The consumer does not repair schemas, synthesize
fields, parse error strings, or choose a compatibility reader.

## Local evidence

Against the P9.8.2 provider checkout, the PSSolver-Control migration produced:

- 71 focused adapter, consumer, import-boundary, provenance, and historical-
  record tests passed;
- 235 tests passed in the complete suite;
- one existing opt-in R5 benchmark skip;
- no failures, xfails, source shadowing, private imports, bytecode cache, or
  pytest cache.

PSSolver's complete source suite also passes with this record: `2607 passed,
8 subtests passed`.  Local installed-wheel packaging and cross-version
checkpoint closure remain P9.8.4 work.

## Authorization boundary

P9.8.3 completes source-consumer migration and makes P9.8.4 planning eligible.
It does not claim cumulative installed-wheel closure, authorize an H100 job,
change production defaults, qualify larger batches, authorize an optimizer
campaign, or complete Phase 9.  The cumulative PDF is not regenerated.
