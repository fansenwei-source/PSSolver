# Phase 9 P9.3: deterministic replay and durable periodic state bridge

Status: `PASS_P9_3_CPU_REPLAY_AND_PERIODIC_STATE_BRIDGE`.

Baseline: `b61e14c9e7e1e67c0afbfc6dfae882a740794005` on
`next/pssolver-v0.2.0-architecture`.

P9.3 qualifies deterministic replay for the CPU execution identity of the
batch-one periodic activity runtime and implements a bidirectional durable
state bridge.  It does not change a production default, stabilize the
provisional package-root API, implement checkpoint scheduling, or claim a
CUDA replay result.  The machine-readable authority is
[phase_9_p93_replay_checkpoint_bridge.json](phase_9_p93_replay_checkpoint_bridge.json).

## Replay contract

For one fixed CPU device, dtype, discretization, and functional execution
identity, repeated calls with the same explicit state, control, and step index
produce bitwise-identical next states and observations.  The result remains
bitwise identical after unrelated calls have used the same runtime object and
after reconstruction of a fresh runtime with the same identity.

The guarantee follows the P9.0 ownership rule: both physical and native
spectral Q representations remain in the flat state tuple; the periodic
Stokes solve has no pressure warm start; Q-gradient reuse and trajectory-
dependent diagnostic caches are disabled; and each functional call replaces
the internal trajectory storage with fresh storage assembled from its input.
Immutable transform plans, wavenumbers, and operator coefficients are allowed
implementation caches because they cannot depend on trajectory history.

The runtime identity now records the replay declaration and checkpoint-bridge
version.  CPU construction advertises `deterministic_replay = bitwise`.
CUDA construction remains fail-closed as `not_qualified` until a later H100
qualification measures that execution identity.

## One periodic checkpoint schema

P9.3 does not introduce a control-project checkpoint format.  Instead it
extracts the existing periodic production checkpoint encoding into
`pssolver.workflows.periodic_checkpoint` and makes both execution paths use
that codec.  The format remains version 1 and retains the existing filenames,
Q component ordering, physical and spectral payloads, integrator progress,
backend identity, and per-file SHA-256 records.

Production capture, atomic write, header inspection, payload load, and restore
are now separate operations.  Restore validates the full target list before
the first production tensor mutation.  A checkpoint is written through a
temporary sibling directory and atomically renamed; an existing destination
is never overwritten.

Functional export adds a small `functional_bridge` metadata object to the
otherwise production-compatible checkpoint.  It binds:

- provisional functional API and bridge format versions;
- functional runtime identity SHA-256;
- production runtime identity SHA-256;
- concrete state-layout metadata and its canonical SHA-256; and
- the periodic batch-one runtime kind.

The production loader deliberately ignores this additive provenance after
validating the ordinary production identities, so a functional export can be
restored directly by the production periodic runtime.  A production-created
checkpoint without the additive object can likewise be imported into the
functional runtime.

## Fail-closed import and autograd boundary

Import first reads only JSON metadata.  Runtime identity, backend contract,
disabled-refresh phase, functional identity, and state layout must agree
before any NPY payload is opened.  The shared codec then validates every
filename, checksum, recorded shape, recorded dtype, and finite value.  Only
after those checks does the bridge construct fresh tensors on the exact target
device without dtype conversion.

Durable export intentionally detaches tensors.  This is not a hidden detach
inside the functional timestep: a durable checkpoint is the serialization
boundary between replay segments.  The independent consumer owns checkpoint
scheduling and builds a fresh one-step graph while replaying each segment.
Gradient correctness and checkpoint-stride gradient invariance remain P9.4.

## Verification and scope

Focused tests establish:

- replay after intervening calls and replay through a fresh runtime;
- exact multi-step replay for spatially varying activity;
- functional export/import byte identity for physical and spectral Q;
- direct production restore of a functional export and bitwise continuation;
- lossless import of a production-created checkpoint;
- identity and layout rejection before `numpy.load` can run;
- checksum, non-finite tensor, cross-identity, and overwrite rejection; and
- preservation of the existing public periodic restart oracle.

The focused P9.0--P9.3 and periodic-restart set passed 46 tests.  The complete
CPU suite passed 2469 tests plus 8 subtests with no failures, skips,
deselections, or expected failures.  `git diff --check` passed.

P9.3 does not qualify CUDA replay, finite-difference or adjoint gradients,
hidden-detach coverage, checkpoint-stride gradient invariance,
production/functional tolerance envelopes, larger batches, H100 performance,
an optimizer, or an independent consumer.  Those remain P9.4 and later.
