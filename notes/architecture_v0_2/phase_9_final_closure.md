# Phase 9 final closure

Status: `PASS_PHASE_9_STABLE_FUNCTIONAL_API_FINAL_CLOSURE`

Phase 9 is complete.  PSSolver exposes the versioned public module
`pssolver.functional.api` at protocol version `1.0`, and the independent
PSSolver-Control project consumes its Periodic and rectangular-Channel
batch-one runtimes from installed wheels without private imports or source
shadowing.  P9.8.5 provides the final cumulative CPU/H100/checkpoint evidence
for Provider commit `0838ecd1cd314a5e8879ce6d1a1ced921d0cf69f` and
Consumer commit `28857a59610df355469dca55372e0d0cd1311f3c`.

## Frozen public boundary

- stable module: `pssolver.functional.api`;
- construction protocol: exact version `1.0`;
- qualified runtime kinds: `periodic_activity_batch_one` and
  `channel_activity_batch_one`;
- checkpoint bridge format version: 1 for both runtime families;
- construction accepts only 1.0, while the exact compatibility reader may
  read the declared `0.1-provisional` checkpoint format;
- silent fallback, heuristic checkpoint repair, private Provider imports, and
  source-worktree shadowing remain forbidden;
- `diagnostics()` is guaranteed by stable factory-returned runtimes as an
  additive API 1.0 facility without expanding the structural protocol;
- compatibility-public aliases keep the deprecation policy frozen in P9.8.2.

The final identity digests are
`426b24fb3226a26f1e58895c2f85bdc3fdc1c25a61e931dc38a334d05751f47d`
for Periodic and
`e64b5291fcf8be62630bd27894b9d8ba5654623e013e324b2ad5eba2133b2e32`
for Channel.

## Qualified behavior

The Periodic runtime includes the per-step self-conjugate-plane Hermitian
projection required for long-horizon finite execution.  The Channel runtime
retains its pure zero-start pressure solve and custom implicit pressure
adjoint.  Production retains its trajectory-dependent pressure warm start.
The canonical zero-start production comparator is the byte-identity oracle;
the real warm-start path has a separate frozen numerical agreement gate.

Both batch-one consumers have installed-wheel identity, replay, restart,
checkpoint, gradient, finite-difference, Taylor, memory, diagnostic, and
negative-gate evidence.  Canonical checkpoint file records and raw-byte
integrity rejection are part of the final closure.

## Version decision

The source remains package version `0.1.2`; this record does not change a
version literal, create a tag, publish a wheel, or promote a production
default.  The completed stable functional boundary is eligible for a
separately reviewed versioned release-candidate preparation.  Release work is
not performed by Phase 9 finalization.

## Explicit non-claims

Phase 9 does not qualify batch sizes above one, Plane control, a production-
scale optimizer, or any scientific-control result.  It does not authorize an
optimizer campaign.  Those are post-Phase-9 projects with their own frozen
contracts and compute qualification.

No PSSolver numerical implementation, PSSolver-Control algorithm,
`nematics3d` installation, shared environment, or historical evidence archive
is changed by this final record.  The architecture archive source list is
updated, but the cumulative PDF is intentionally not regenerated.

Local finalization checks pass: the complete Provider Phase 9 test set reports
`159 passed`, and the Consumer complete suite reports `244 passed, 1 skipped`.
The only skip is its existing opt-in R5 benchmark.  Both repositories pass
`git diff --check` and strict JSON parsing.
