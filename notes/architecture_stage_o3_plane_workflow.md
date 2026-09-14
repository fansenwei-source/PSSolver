# Architecture Stage O.3: shared Plane workflow and restart boundary

## Outcome

Stage O.3 moves step coordination, canonical Q/u/p observations, diagnostics,
checkpointing, metadata completion, and the `COMPLETE` marker behind one Plane
workflow shared by `legacy_production` and `separated_canary`.

It does not change the default runtime, promote the canary, modify Channel or
the generic solver, or claim cross-backend checkpoint compatibility.

## Observation and completion contract

Both runtime adapters expose the same field, step-clock, synchronization,
normal-force diagnostic, flow-diagnostic, and restart surfaces.  The workflow
writes the established files:

- `Q_<step>.npy` with shape `(Nx, Ny, Nz, 5)`;
- optional `u_<step>.npy` with shape `(Nx, Ny, Nz, 3)`;
- optional `p_<step>.npy` with shape `(Nx, Ny, Nz)`;
- the existing structured `diagnostics.npy` and equivalent CSV columns.

Temporary snapshot files are replaced atomically and existing observations are
never overwritten.  On success, complete metadata is written first and
`COMPLETE` is deliberately the final write.  An interrupted workflow therefore
cannot advertise completion merely because a stale marker was created early.

## Versioned same-backend checkpoint

The Stage O.3 checkpoint format stores:

- the explicit runtime path and a SHA-256 of numerical runtime identity;
- absolute completed steps and the exact spectral-refresh phase;
- all five evolved Q components in both spatial and native spectral storage;
- the backend restart contract (currently stateless for both Plane algebraic
  implementations);
- shape, dtype, finiteness, and per-file SHA-256 records.

Runtime identity binds geometry, boundary conditions, spectral/numerical
configuration, physical model/preset, timestep, refresh policy, zero-mode
policy, precision controls, and runtime path.  It deliberately excludes the
fresh-run initializer, diagnostics, output directory, requested additional
steps, save cadence, checkpoint cadence, and restart source so a continuation
can use a new workflow envelope around the checkpoint-supplied evolved state.

`--restart-from` always creates a distinct output directory and interprets
`--steps` as additional steps.  Header validation rejects a runtime-path or
runtime-identity mismatch before output creation, initial-condition generation,
or solver construction.  No implicit cross-format conversion exists.

## Characterization requirements

O.3 tests require:

- unchanged filenames, array shapes/dtypes, and final marker order;
- checksum rejection for modified checkpoint tensors;
- byte-exact continuous versus split restart for each runtime path, including
  a split inside the spectral-refresh interval;
- finite, schema-identical diagnostics on both paths;
- early cross-backend rejection with no output directory created;
- no workflow import or promotion in Channel, generic `SpectralSolver`, or the
  top-level `pssolver` API.

Stage O.4 remains responsible for bounded CPU characterization and one balanced
H100 legacy/canary qualification.  O.3 alone does not authorize changing the
production default.

## Local qualification

The completed implementation passed 877 tests plus 8 subtests.  Small CPU
production runs also confirmed that the O.3 legacy workflow remains byte-for-
byte identical to the O.2 baseline for every saved initial/Q/u/p array and for
both diagnostic files.  Continuous and split runs were byte-exact at the final
step on both runtime paths, including a checkpoint taken inside a spectral-
refresh interval.  H100 characterization remains intentionally deferred to
Stage O.4.
