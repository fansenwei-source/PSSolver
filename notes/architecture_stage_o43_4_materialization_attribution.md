# Architecture Stage O.4.3.4: remaining materialization attribution

## Decision from Stage O.4.3.3

The producer-owned H/stress layout was scientifically equivalent and reduced
projected-transform copy/cat assembly from eight to five batches per steady
timestep. Its R320 candidate/control mean timestep ratio was
`0.982766967768`: a stable improvement in all three pairs, but short of the
prospectively frozen `0.98` gate. R128 also regressed slightly and R320 peak
reserved memory increased. Stage O.4.3.3 therefore remains `B_neutral`; it
does not authorize O.4.4 or a production promotion.

Stage O.4.3.4 is a measurement-only follow-up. It does not reinterpret the
O.4.3.3 gate and does not introduce another tensor layout.

## Attribution boundary

The projected-transform scheduler records a semantic source for every batch
assembly while bounded performance instrumentation is enabled. Sources are
assigned at the nearest existing execution boundary:

- algebraic system physical dependencies;
- algebraic system spectral outputs;
- explicit-RHS transient dependencies;
- explicit-RHS spectral outputs.

For every source the scheduler reports integer batch/component counts, logical
input bytes, exact copy/cat output bytes requested, contiguous-view counts,
fallback reasons, and the semantic timing-region name. A copy/cat output has
the same number of bytes as its concatenated logical inputs; a view reports
zero materialized output bytes. These cumulative byte counts are not
misrepresented as CUDA peak memory or retained storage.

The timing recorder uses deferred CUDA events and synchronizes only when the
profile snapshot is requested. Source records contain no tensors, storage
objects, or cross-generation references. The normal production path does not
enable this instrumentation.

## Bounded H100 diagnostic

The diagnostic reuses the unchanged O.4.3.3 `boundary_packed` candidate and the
frozen R320 production reference. It runs three repetitions with ten warm-up
steps and twenty measured steps. There is no new trajectory comparison because
this commit changes only diagnostics; the O.4.3.3 100-step numerical evidence
remains immutable and is bound by SHA-256.

The analysis requires:

- all projected batches to have a semantic source;
- identical count structure across the three trials;
- zero retained tensor references;
- finite timing and byte metrics;
- unchanged production and initial-Q identities.

The report ranks remaining copy/cat sources by measured assembly time and
reports a non-authoritative screening flag when the leading source consumes at
least three percent of the measured timestep. This flag only identifies a
target worth reviewing. It cannot authorize implementation, O.4.4, a default
change, or production promotion.

## Isolation

All additions remain under `pssolver.experimental`, `benchmarks`,
`scripts_plane`, tests, and notes. Plane production, Channel, the generic
solver, equations, transform order, field lifetimes, tensor layouts, and every
default remain unchanged.
