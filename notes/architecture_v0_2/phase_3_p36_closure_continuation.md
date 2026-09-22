# Phase 3 P3.6 cumulative closure continuation

Status: `FAIL_P3_6_CLOSURE_HARNESS_RECOVERY_PENDING`.

The cumulative Phase 2-baseline versus Phase 3-candidate trajectory and
restart gates are now complete.  One H100 job ran all eight 100-step
continuous trajectories and all four bidirectional R128 cross-version
restart trajectories.  Parent and candidate Q/u/p were byte-identical for
both `legacy_production` and `separated_canary` at R128 and R320, both R128
runtime paths matched their frozen oracles, and every split restart matched
its corresponding continuous output byte-for-byte.

The job then stopped at the reset/rebind gate before any negative gate ran.
The external qualification harness passed a hand-built, incomplete
`production_metadata` object to the separated-canary constructor.  It
contained configuration and runtime-selection records but omitted the
required Stage K comparison contract.  The resulting error is a harness
input-contract failure, not a solver, CUDA, numerical, scientific, or
performance failure.

The eight completed continuous trajectories and four completed restart
trajectories are immutable reusable evidence.  The next recovery is limited
to constructing reset/rebind inputs from authoritative production metadata,
completing the reset/rebind GPU smoke, and running the eleven pre-registered
negative gates.  It must not repeat CPU suites, package qualification,
profiling, simulation-timer adjudication, continuous trajectories, or
restart trajectories.

Phase 3 remains open.  The production default remains `legacy_production`,
`separated_canary` remains an explicit non-promoted canary, and Phase 4 is
not authorized.

The machine-readable record is
[phase_3_p36_closure_continuation.json](phase_3_p36_closure_continuation.json).
