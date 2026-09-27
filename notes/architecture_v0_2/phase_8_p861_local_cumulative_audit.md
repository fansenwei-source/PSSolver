# P8.6.1 local cumulative closure audit

## Status

P8.6.1 is complete.  The repository-owned fail-closed analyzer binds all five
authoritative Phase 8 prerequisite records, compares the live and isolated
installed-wheel public catalogs, verifies the eight-case rejection matrix,
and confirms that no runtime default or public capability was promoted.

Classification:

`PASS_P8_6_1_LOCAL_CUMULATIVE_AUDIT`

This is a local CPU/package qualification.  It does not claim that P8.6.2 has
run, does not close Phase 8, and does not authorize Phase 9.

## Frozen evidence

The audit implementation is commit
`aeee095a2191fe80b311e488144566f55da3c19f`.  It validates the authoritative
P8.1, P8.2, P8.3, P8.4.5, and P8.5.6 JSON records using their frozen SHA-256
identities from the P8.6.0 plan.

The analyzer is:

`benchmarks/analyze_phase8_cumulative_closure.py`

Its SHA-256 is:

`cd68cdd36edef4c65ff83dbbef78012d061e72c9e19fc45a3e54a66be2b33ce8`

The audit test is:

`tests/test_phase8_p861_local_cumulative_audit.py`

Its SHA-256 is:

`2a01d48977366710c49b27dce84cfb4b7eb9d0910ae10f9bc9f93d9ae762c20e`

## Installed-wheel audit

The analyzer copied only packaging inputs and the current `pssolver` package
into an isolated temporary source tree, built one wheel without dependencies
or build isolation, extracted it outside the checkout, and imported from the
extracted wheel.  The observed wheel contained 232 members, the import was not
shadowed by the source checkout, and every required public root symbol was
present.

The particular local artifact was
`pssolver-0.1.2-py3-none-any.whl`, with SHA-256
`4803b8c80e05f20e564fe8a121cfb7b38f7565d47209db77a3abc1d1496b23fb`.
This digest identifies this audit artifact; P8.6.2 must build and record its
own wheel from its frozen source commit rather than assume byte-reproducible
ZIP timestamps.

The source and installed-wheel capability catalogs were exactly equal:

- two public models;
- three public geometries;
- eight public boundary policies;
- four qualified model/geometry combinations;
- six qualified runtime paths.

Robin remains declarable but has no public executable application.  Plane and
Channel defaults remain `legacy_production` and `legacy_channel`.

## Rejection matrix

All eight P8.6.0 rejection cases are bound to executable repository test nodes:

1. unregistered model/geometry;
2. Robin Q in a public complete timestep;
3. prescribed Q on an unqualified geometry;
4. wall data on a periodic face;
5. geometry-incompatible velocity or pressure policy;
6. prescribed nonzero Neumann flux in the current Robin operator slice;
7. dynamic, spatial, callable, or trainable Robin data outside the static contract;
8. checkpoint identity or payload mismatch.

The complete CPU suite executes those nodes rather than treating source-text
presence as scientific evidence.  The analyzer additionally fails closed if
any node is renamed, removed, or no longer bound to the frozen matrix.

## Verification

- P8.6.1 plus P8.6.0 targeted tests: 12 passed.
- Full CPU suite: 2410 passed, 8 subtests passed.
- Failures, skips, deselections, and xfails: zero.
- `git diff --check`: passed.
- The existing verbatim archive PDF was not regenerated or modified.

## Scope and next authorization

P8.6.1 changed only the repository analyzer and its tests.  It did not change
the timestep, numerical operators, compiler registry, public declarations,
checkpoint/output schemas, or production defaults.

P8.6.2 planning is now eligible.  P8.6.2 still requires a separately bound
source commit and one installed-wheel H100 job.  P8.6.3 and Phase 9 remain
unauthorized.
