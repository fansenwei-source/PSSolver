# Phase 8 P8.3: complete-stress Beris--Edwards in a rectangular Channel

Status: `P8_3_OBSERVATION_SYNC_RECOVERY_LOCAL_CANDIDATE_H100_REQUIRED`.

Baseline: `b753d24acec1245f6335dd45ed5cb1f5d8fefd19` on
`next/pssolver-v0.2.0-architecture`, after the completed P8.2 periodic H100
closure.

P8.3 registers the fourth public model--geometry application:

```text
complete_stress_beris_edwards + rectangular_channel
```

It uses Q Neumann conditions on both bounded axes and no-slip velocity on
both bounded axes.  The existing legacy Channel model, Plane application,
periodic application, and all production defaults remain unchanged.

## Independent two-bounded-axis force lowering

P8.3 does not reuse the Plane one-bounded-axis distortion split.  With
`x` periodic and `y,z` bounded, each row-major distortion-stress component
has its own tensor-product parity.  The `xy/yx`, `xz/zx`, and `yz/zy`
components respectively use P/D/N, P/N/D, and P/D/D bases; diagonal
components use P/N/N.

The new internal component-basis divergence operator transforms and
differentiates every stress component in its native basis.  Derivative terms
are combined in physical space because a row's three derivatives generally
do not share one spectral basis.  The complete force is then projected into
the P/D/D no-slip velocity space.  An analytic manufactured test independently
checks this derivative and projection path to float64 tolerance.

## Architecture and solver ownership

The active-nematic model package owns only constitutive kernels and
tensor-free basis declarations.  The concrete composition of complete stress,
the Channel Schur/PCG solver, projector, integrator, and legacy spectral engine
is owned by `pssolver.runtime.channel_beris_edwards`.  This preserves the
model/geometry separation and avoids a model-to-linear-solver dependency.

The runtime reuses the already extracted no-slip Channel pressure solver:

- velocity and force basis: periodic/Dirichlet/Dirichlet;
- pressure basis: periodic/Neumann/Neumann;
- pressure gauge: zero mean;
- matrix-free Schur solve: PCG with explicit warm-start state;
- uniform velocity nullspace: absent because two axes are no-slip.

The complete-stress public constructor still records its historical
`zero_mean` tangential declaration.  Lowering records that declaration but
resolves the effective Channel velocity-nullspace policy to `not_applicable`.
This is not treated as a pressure gauge and does not remove a physical Channel
mode.

## Public execution and exact restart

The public compiler accepts only the exact registered runtime path
`channel_complete_stress`, full-complex storage, TF32 off, projected Euler,
disabled spectral refresh, a finite external Q snapshot, physical-space
stress-divergence summation, and an explicit warm-start PCG contract.  Unknown
or unsupported choices fail before allocation; runtime fallback is forbidden.

The workflow records Q/u/p observations, pressure residual diagnostics,
metadata, and an exact checkpoint.  Checkpoints contain spatial and spectral
Q/u/p state, pressure-PCG warm-start state, tensor hashes, shape, dtype,
runtime identity, backend identity, and integrator progress.  Validation of
all records finishes before target-state mutation.  On CPU, a six-step
continuous trajectory and a three-step plus three-step restart are byte-for-byte
identical for Q, velocity, and pressure; pressure-state byte tampering is
rejected before advancement.

## Local result and remaining gate

The P8.3 implementation, focused architecture tests, old Plane/Channel/P8.2
regressions, manufactured operator test, finite trajectory, exact restart,
and tamper rejection pass locally:

```text
focused P8.3 recovery and architecture suite: 64 passed
complete CPU suite: 2243 passed, 8 subtests passed
git diff --check: pass
```

No test failed, skipped, xfailed, or was deselected in this local environment.

P8.3 is not complete until a frozen H100 qualification verifies clean
installed-package identity, CUDA-only tests, analytic force projection,
R128/R320 performance and memory non-regression, transform-call accounting,
finite production trajectories, exact restart, negative checkpoint gates,
immutable provenance, and checksums.

Until then no runtime is promoted, no default changes, P8.4 and Phase 9 are
not authorized, and no scientific benchmark claim is made.

## Initial H100 qualification and restart recovery

The first H100 qualification (Job 10842705) passed the installed-package,
CUDA, manufactured-force, Channel PCG, finite-state, memory, and all six
profile gates.  It stopped only because the C128 and C512 split/restart
trajectories differed from their continuous references at roundoff scale
instead of being byte-for-byte identical.  The largest reported relative-L2
differences were approximately `1.5e-16` for Q, `5.8e-15` for velocity, and
`1.8e-14` for pressure.  These values are not a scientific failure, but the
exact-restart software contract was deliberately not relaxed.

The first recovery reconstructed the single-use Q-gradient cache from restored
Q spectra.  This remains correct derived-state hygiene and avoids checkpointing
fifteen redundant gradient arrays.  Its focused H100 run (Job 10842781),
however, reproduced the original differences exactly and localized the first
u/p divergence before the resumed advance.  It therefore disproved the cache
as the primary cause.

The remaining cause was observation synchronization.  Diagnostics, saves,
checkpoints, and final output could recompute the Channel static solve and
mutate the PCG warm start without marking the static fields current.  The next
timestep then solved the same state again.  A continuous workflow and a split
workflow could consequently execute different numbers of PCG solves at the
split boundary even though Q and every persistent checkpoint tensor matched.

The second recovery makes observation synchronization idempotent.  It solves
only when the static state is stale, marks the result current, and lets the
next timestep consume that already-computed state and its published gradient
cache.  New tests prove that dense versus sparse diagnostics/save schedules
produce byte-identical Q/u/p and that there is exactly one static solve per Q
state.  Together with the longer restart oracle, the focused recovery suite
passes 64 tests and the complete CPU suite passes 2243 tests plus 8 subtests.
Final closure still requires a focused H100 rerun of restart and negative
gates; previously passed manufactured-force and profiler evidence should be
reused rather than repeated.
