# Phase 3 P3.6 reset/rebind recovery v3

Status: `FAIL_P3_6_LEGACY_POST_RESET_HARNESS_DIAGNOSTIC_COMPLETE`.

Recovery v3 passed all 27 synthetic tests and every immediate legacy reset
contract, then stopped in the aggregate aligned post-reset evolution gate.
The external harness raised before persisting its per-predicate results, so
the H100 archive cannot identify the exact false predicate.  This is a
qualification-harness observability failure; it is not sufficient evidence of
a solver or numerical failure.

A local CPU diagnostic reproduced the likely legacy mismatch.  Aligning a
fresh control's progress through the runtime adapter marks its static fields
as current.  If the harness does this without first synchronizing the fresh
control's algebraic/static fields, the first post-reset step skips the static
recompute and diverges from the reset runtime.  The observed spatial maximum
difference was `1.5515657882568605e-05`.  Following the canonical checkpoint
restore order instead---reset independent fields, call
`synchronize_for_observation()`, then call `restore_progress()`---restored
bitwise equality of both spatial and spectral state.  The same diagnostic on
the separated canary remained bitwise equal with and without the explicit
sync, but the production restore order should still be used uniformly.

The next recovery must persist each subgate result atomically before any
aggregate assertion and must use the production checkpoint restoration order.
No previously qualified performance, trajectory, or restart gate needs to be
repeated.

The machine-readable record is
[phase_3_p36_reset_rebind_recovery_v3.json](phase_3_p36_reset_rebind_recovery_v3.json).
