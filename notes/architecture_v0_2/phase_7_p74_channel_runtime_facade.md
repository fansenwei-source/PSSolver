# Phase 7 P7.4: opt-in Channel runtime facade and exact restart

Status: `PASS_P7_4_CHANNEL_RUNTIME_FACADE`.

P7.4 keeps the P7.1 Channel run specification at its explicit geometry
submodule and adds a package-owned runtime-selection facade.  The frozen
`pssolver.configuration` root export list remains unchanged.  `legacy_channel`
remains the default, while
`compiled_channel_v2` is available only when it is explicitly requested.
Both solver constructors remain application-owned injection points in this
slice, so the runtime layer does not depend on the legacy `pssolver.channel`
facade or on the experimental compiled implementation.  A missing or failing
selected builder raises immediately; there is no fallback between paths.

The runtime request verifies that configuration identity and runtime-selection
metadata came from the same immutable `ChannelActiveNematicRunSpec`.  The run
spec now exposes separate canonical, runtime, and requested/effective
identities.  Runtime identity includes geometry, boundary spaces, numerical
policy, material and Stokes parameters, PCG controls, precision, batch size,
runtime path, and timestep.  It excludes device, output locations, run length,
save schedule, diagnostics schedule, and initial-condition provenance so that
a compatible continuation may choose a different workflow without weakening
the numerical identity gate.

Both runtime adapters expose the same narrow surface:

- selected runtime identity and completed-step count;
- solver and field ownership;
- stable zero-copy Q, velocity, and pressure output views;
- stepping with the historical callback contract;
- observation synchronization;
- pressure warm-start capture and restore;
- backend restart identity; and
- refresh-clock restoration.

The version-1 Channel checkpoint is intentionally geometry-specific.  It
stores the five evolved Q components and all four algebraic `u/p` components
in both physical and spectral representations, the PCG pressure warm-start,
completed-step and spectral-refresh counters, runtime identity, and backend
restart metadata.  Persisting the synchronized algebraic state is necessary
for byte-identical continuation: recomputing the Schur solve during restore
would introduce an extra PCG application and could change the rounding path.

Checkpoint directories are written atomically without pickle.  Each array has
shape, dtype, and SHA-256 metadata.  Loading rejects missing, renamed,
non-finite, shape-inconsistent, dtype-inconsistent, or checksum-mismatched
arrays.  Restore rejects cross-runtime checkpoints, runtime-identity changes,
backend-contract changes, and tensor corruption before mutating the target.
Only same-runtime restart is authorized.

CPU qualification establishes:

- unchanged `legacy_channel` default and explicit `compiled_channel_v2`
  selection;
- fail-closed builder selection with no fallback;
- zero-copy and stable Q/u/p output ownership on both paths;
- five-step byte-for-byte legacy/compiled trajectory identity through the
  formal facade;
- versioned disk checkpoint round trips for both paths;
- byte-for-byte continuous versus split-restart identity for both paths;
- early cross-runtime and tamper rejection; and
- unchanged `Channel.py` and `pssolver.channel` source identities.

The geometry-specific package runtime facade is now real, but the historical `Channel.py` entry
point remains byte-identical and does not consume the new run spec.  P7.4 does
not add an application runner, change the production default, qualify an
installed wheel, authorize H100 execution, or add a boundary law.  Those
closure and connection questions remain P7.5/P7.6 gates.

Local validation passed 100 targeted and compatibility tests.  The complete
suite passed 2,025 tests plus 8 subtests with no failure.

P7.4 makes P7.5 planning eligible but does not authorize its implementation.

The machine-readable result is
[phase_7_p74_channel_runtime_facade.json](phase_7_p74_channel_runtime_facade.json).
