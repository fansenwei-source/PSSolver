# PSSolver 0.2.0rc6 release candidate

PSSolver 0.2.0rc6 is the public-error, validation, checkpoint-control-flow,
and restart-atomicity follow-up to the immutable `v0.2.0rc5` prerelease.
The planned implementation queue RC6-1 through RC6-7 is complete at source
level; this record begins cumulative qualification of the frozen candidate.

## Closed implementation queue

- RC6-1 gives Plane `legacy_production` and `compiled_v2` one public
  non-finite diagnostic exception contract.
- RC6-2 aligns Channel tolerance success with the projected recurrence
  residual, uses a conservative dtype-derived floor, and rejects unsupported
  sub-floor declarations during compilation.
- RC6-3 validates complete-stress model parameters and external Q component
  layout before runtime construction or output materialization.
- RC6-4 makes Channel pressure failure public and atomically records failed
  started runs for Plane and legacy Channel.
- RC6-5 preserves non-reentrant checkpoint early-stop control flow through
  the functional public facade.
- RC6-6 prevents rejected restarts from materializing a missing target output
  directory before checkpoint validation succeeds.
- RC6-7 closes the independent PSSolver-Control Periodic stale-pin
  guard-order regression without changing either project runtime.

## Stable public boundary

The package version changes to `0.2.0rc6`.  Functional API version 1.0,
adapter versions, checkpoint schemas, the four qualified model--geometry
combinations, batch-one functional scope, and production defaults do not
change.  Compiled runtimes remain opt-in.  Lifting remains fail-closed.
PSSolver-Control remains an independent consumer, and `nematics3d` remains an
unmodified optional analysis dependency.

## Cumulative qualification boundary

This source freeze is not a cumulative qualification PASS.  The frozen source
must next pass the full source CPU suite, a fresh installed-wheel CPU suite,
the released-checkpoint compatibility and tamper gates, Provider-identity
impact review, any required independent PSSolver-Control migration and
installed-wheel qualification, and one separately authorized single-H100
cumulative run.

No push, tag, GitHub release, PyPI publication, default promotion, optimizer
campaign, scientific-control conclusion, or H100 submission is authorized by
this record.
