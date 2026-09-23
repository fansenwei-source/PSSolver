# Phase 7 P7.5: Channel local package and restart closure

Status: `PASS_P7_5_CHANNEL_LOCAL_CLOSURE`.

P7.5 connects the P7.4 Channel runtime facade to a package-owned application,
observation layer, and finite-run workflow.  The historical `Channel.py` and
`pssolver.channel` remain byte-identical rollback oracles.  The implicit
runtime remains `legacy_channel`; `compiled_channel_v2` is explicit opt-in and
never falls back.

The application resolves one immutable `ChannelActiveNematicRunSpec`, creates
the generated Q initial condition, injects exactly one selected builder, and
hands the adapter to a backend-neutral workflow.  The compiled implementation
is reached through one named runtime bridge, preserving the rule that
applications do not import experimental implementations.  Snapshot files from
the historical script are not silently reinterpreted as workflow checkpoints;
the P7.5 application accepts the generated initial condition and versioned
same-runtime Channel checkpoints, while legacy snapshot handling remains in
`Channel.py`.

The workflow preserves the established file layout: `Q_<step>.npy` has five
components on the final axis, `u_<step>.npy` has three, and `p_<step>.npy` is a
scalar grid.  It writes the divergence, PCG residual, and both bounded-axis
wall-normal momentum diagnostics in the established structured NPY and CSV
schema.  Observation and diagnostic writes refuse overwrites.  Metadata is
written as running and complete, and `COMPLETE` is the final successful write.

The run declaration now carries optional `checkpoint_interval` and
`restart_from` workflow controls.  They are excluded from runtime identity.
Restart restores Q and algebraic u/p in physical and spectral form, the PCG
pressure warm-start, spectral-refresh counters, and completed-step clock.
Only the exact same runtime path and runtime identity are accepted.  Cross-
runtime, checksum, shape, dtype, non-finite, backend-contract, and identity
changes fail before target mutation.

Local qualification established:

- one-step and 100-step legacy/compiled `Q/u/p` byte identity;
- valid finite Channel diagnostics and explicit requested/effective metadata;
- continuous versus 7+13 split restart byte identity for both runtime paths;
- pressure warm-start persistence through the versioned disk checkpoint;
- cross-runtime restart rejection and no false `COMPLETE` marker;
- refusal to overwrite nonempty output directories;
- fail-closed progress and legacy-snapshot contracts;
- unchanged `Channel.py` and `pssolver.channel` identities;
- 58 P7 targeted tests, followed by the complete suite with 2,034
  passing tests and 8 passing subtests; and
- a fresh installed wheel, imported outside the checkout, completing one CPU
  step through both runtime paths with Q/u/p, diagnostics, metadata, and
  `COMPLETE`.

The qualified wheel was `pssolver-0.1.2-py3-none-any.whl`, SHA-256
`685f2cd62bff480474b53ea94cb892c9ea5c152d529a32a6ff67611dbc710b85`.

P7.5 is locally complete.  P7.6 H100 planning is eligible, but this record does
not authorize an H100 job, a long Channel run, default promotion, a new boundary
law, or any modification of Plane evidence.

The machine-readable record is
[phase_7_p75_local_closure.json](phase_7_p75_local_closure.json).
