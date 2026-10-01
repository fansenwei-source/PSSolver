# P9.8.5 stable functional API cumulative closure

Status: `PASS_P9_8_5_STABLE_FUNCTIONAL_API_CUMULATIVE_CLOSURE_WITH_CANONICAL_ZERO_START_CHANNEL_ORACLE`

P9.8.5 qualifies the installed-wheel functional API 1.0 consumers against
PSSolver commit `0838ecd1cd314a5e8879ce6d1a1ced921d0cf69f` and
PSSolver-Control commit `28857a59610df355469dca55372e0d0cd1311f3c`.
The cumulative record combines the qualified long-horizon Periodic Hermitian
repair, the Periodic and Channel consumer gates, and the schema-aware
checkpoint-integrity recovery.  The final CPU-only recovery contains a
`COMPLETE` marker and verifies all 30 manifest entries.

## Periodic component

The repaired Periodic packed half-spectrum projects every self-conjugate
plane back into the Hermitian subspace after each step.  Both frozen
long-horizon cases remain finite through time 200 with zero measured
Hermitian violation.  The focused paired H100 adjudication passed timing and
memory non-regression, and the P9.8.5 consumer evidence passed identity,
checkpointed/full-history gradient, finite-difference, Taylor, checkpoint,
memory, and negative gates.  Its final public identity digest is
`426b24fb3226a26f1e58895c2f85bdc3fdc1c25a61e931dc38a334d05751f47d`.

## Channel component and pressure semantics

The Channel public identity digest is
`e64b5291fcf8be62630bd27894b9d8ba5654623e013e324b2ad5eba2133b2e32`.
Gates A through F pass, as do its checkpoint, gradient, Taylor, bounded-memory,
and negative gates.

The production and functional paths intentionally use different pressure-PCG
initial guesses.  Production retains its trajectory-dependent warm start;
the pure functional map uses `zero_every_call` and neither reads nor writes
production warm state.  The qualification therefore separates two contracts:

1. a production comparator explicitly zeroed before every pressure solve is
   byte-identical to the functional path for physical Q, spectral Q, velocity,
   and raw pressure over the frozen one-step and four-step checks;
2. the unmodified warm-start production path satisfies the frozen numerical
   agreement bounds against the functional path.

This is not a relaxed equality test.  Byte identity is applied to paths with
the same algorithmic initial-guess semantics, while the actual production
path retains a separately bounded comparison.  `pressure_guess` is not added
to functional state, so replay, checkpointing, and differentiation remain
pure and deterministic.

## Checkpoint integrity

The final recovery reads the Periodic canonical checkpoint metadata instead
of assuming a monolithic `state__q_physical.npy` file.  It discovers the ten
component payloads, selects `spatial.Qxx` from the canonical records, changes
one payload byte in an isolated copy, and observes the expected checksum
rejection before target mutation or timestep execution.  The source
checkpoint, metadata, and file length remain unchanged.

## Evidence chain

| Evidence | Manifest result | Manifest SHA-256 |
|---|---:|---|
| Periodic repaired-provider focused H100 qualification | 36/36 PASS | `158390775fb371aa4bd8a2d4232d8607e7ab73f2b3c04883349a42a3d0a21206` |
| P9.8.5 repaired-provider cumulative recovery v9 | 230/230 PASS | `6e544074c5f6b5fc0c02f6e84eb1563f78b41131e705b61e80f668fdfb816ccc` |
| Channel pressure-causality analysis v11 | 29/29 PASS | `3cfd6a17bf1c67cfc0a47dfa4415ce10ab9cd48353b119d0846283f5d6a0f06c` |
| Channel canonical-zero-start closure v12 | 72/72 PASS | `49d9a625dceaac20dacd2a219dd97a8d9902535df93429fbe5a0680f71b5bb5a` |
| Schema-aware CPU finalization v13 | 30/30 PASS | `179c167279303cd54793f5cb7792cdb13f643662fe6bef8a5055b2492f67035b` |

The v12 H100 job is retained as a failed external-helper archive: all Channel
science gates passed before its hard-coded checkpoint filename failed.  The
v13 recovery performs no scientific timestep and submits no Slurm job; it
repairs only the external evidence harness and completes the cumulative
analyzer.

## Boundary

P9.8.5 qualifies only the two batch-one functional consumers and stable API
1.0 contract.  It does not change a production default, qualify a larger
batch, authorize an optimizer campaign, or report a scientific-control
result.  P9.8.6 owns the final Phase 9 record.  The cumulative PDF is not
regenerated here.
