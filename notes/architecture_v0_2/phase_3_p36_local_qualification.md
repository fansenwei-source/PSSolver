# Phase 3 P3.6 local closure qualification

Status: `PASS_P3_6_LOCAL_H100_CLOSURE_PENDING`.

Phase 2 closure baseline:
`0d5b7186b7a104354db173e04a43e68c15daa515`.

Phase 3 cumulative candidate:
`227b095c12e494ca74756228a30962692474c516`.

The local half of P3.6 is complete.  It qualifies the cumulative Phase 3
change, rather than only the final P3.5 delta.  The candidate is now frozen
for one final balanced H100 closure task; no runtime is promoted and no
production default changes as a consequence of this local result.

The machine-readable record is
[phase_3_p36_local_qualification.json](phase_3_p36_local_qualification.json).

## Source and CPU gates

The Phase 2 baseline is an ancestor of the candidate.  Their cumulative diff
contains 23 reviewed files: the Phase 3 ADR and evidence, the disconnected
state/workspace contracts, the shared step program and state-backed
integrator, the canary and legacy connection edges, and their tests.
`git diff --check` passed.

The complete CPU suites passed in the same local environment:

| Revision | Tests | Subtests | Failures |
|---|---:|---:|---:|
| Phase 2 closure baseline | 1623 | 8 | 0 |
| Phase 3 cumulative candidate | 1654 | 8 | 0 |

The environment used Python 3.12.13, PyTorch 2.6.0, NumPy 2.4.3, SciPy
1.16.0, and pytest 9.1.1.

## Cumulative numerical and restart gates

Both revisions ran four CPU float64 eager steps at 8 x 8 x 8 through both
`legacy_production` and `separated_canary`.  Parent/candidate continuous
Q/u/p, initial-condition artifacts, and completion state were byte-identical.
For each runtime, baseline-step-2 to candidate-step-4 and candidate-step-2 to
baseline-step-4 restarts matched the corresponding continuous output
byte-for-byte.

The final hashes remain the Phase 2 closure oracles:

| Runtime | Output | SHA-256 |
|---|---|---|
| legacy | `Q_4.npy` | `ba2165de06c553b7c779bde82ab95af8a3a32a07098e59b7cccb13651a0ba35d` |
| legacy | `u_4.npy` | `452a4410f7e317abce99357446a187edd90fa70d04c4ca9571490679a894ef9e` |
| legacy | `p_4.npy` | `59371ae09916749be610cd3837bdb1c83e56c9606d6971bb59ece9535c0874a4` |
| canary | `Q_4.npy` | `f645679f1c562ee5dfa50adf2d741dac44ccf190dd5c4993c6d893659fe0b6fe` |
| canary | `u_4.npy` | `9035a1ad5384b50190855e9e8149fb0f37722ff2319619bddd2068e45d644e17` |
| canary | `p_4.npy` | `efe175d6033a77c36a3d6f0674fc825b6f9a9c15b3ee0694fd438095f152f05a` |

## Package gate

A clean source archive of the candidate produced a fresh sdist and wheel.
The wheel was installed into a new environment outside the source checkout.
The installed package, rather than the checkout, supplied the runtime-state,
workspace, state-backed integrator, step-program, and Plane runtime modules.
The console dry-run created no output directory, and one installed CPU step
through each runtime produced finite float64 Q/u/p with a valid `COMPLETE`
marker.

| Artifact | SHA-256 |
|---|---|
| `pssolver-0.1.2-py3-none-any.whl` | `b05694c62a28fadb721c1fd7e61b8d93650fefa3f012665158ce02c7d7dc6f6f` |
| `pssolver-0.1.2.tar.gz` | `ae0b832c6615d9eb616f1cd509c62b2666f881ea4fa19d452aaabc5f95e00854` |

These artifact hashes record this local build; the final H100 task must build
and inspect fresh artifacts in its own environment rather than require
byte-identical archives across packaging environments.

## Final H100 closure gate

P3.6 is not complete until one cumulative Phase 2-baseline versus Phase
3-candidate H100 task passes all of the following:

- complete baseline and candidate CPU suites plus both CUDA-only tests;
- clean candidate sdist/wheel and isolated installed-runtime smoke;
- three balanced profiler trials per role at R128 and R320;
- mean/median timestep and peak allocated/reserved ratios no greater than
  1.02;
- seven forward and 32 inverse transforms per step, with zero graph break,
  compile fallback, projected-transform fallback, OOM, NaN, Inf, or CUDA
  error;
- 100-step parent/candidate continuous trajectories for both runtime paths at
  R128 and R320, with same-runtime byte-identical Q/u/p and frozen R128 oracle
  matches;
- bidirectional same-runtime cross-version split restart at R128;
- format-v1 checkpoint, reset/rebind, tensor-tamper, metadata/hash-tamper,
  runtime-identity mismatch, and cross-runtime-restart gates;
- unchanged public API, schema-v1 metadata, output schedule, production
  default, and non-promotion of `separated_canary`.

Passing this gate closes Phase 3.  It does not merge into `develop`, promote a
runtime, change a default, or begin Phase 4 without a separate decision.
