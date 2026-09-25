# Phase 8 P8.2: complete-stress Beris--Edwards in a periodic box

Status: `P8_2_LOCAL_CANDIDATE_H100_QUALIFICATION_REQUIRED`.

Baseline: `f8789461ee71391885d5e04c6906a13bc83a26b0` on
`next/pssolver-v0.2.0-architecture`, with the uncommitted P8.0 and P8.1
capability records retained in the same working tree.

P8.2 implements the first new executable model--geometry combination after
Phase 7:

```text
complete_stress_beris_edwards + periodic_box
```

The local equation, architecture, trajectory, and restart gates pass. This is
not yet a completed qualification: installed-package evidence and the frozen
H100 performance, memory, transform-call, runtime-identity, and production
trajectory gates remain mandatory. Plane and Channel production defaults are
unchanged.

## Periodic Stokes operator

`PeriodicModalStokesSolver` is a direct all-Fourier Stokes--Brinkman solve.
For every nonzero wave vector it applies the Helmholtz projection and solves
the scalar pressure Schur relation. It supports both full-complex and
Hermitian-half spectral storage.

Pressure and velocity nullspaces are separate contracts:

- pressure always uses a zero-mean gauge;
- `zero_mean` with zero friction removes the complete uniform force and
  velocity mode in `ux`, `uy`, and `uz`;
- `friction` requires positive drag and retains the uniform velocity, which
  is then determined by force divided by friction.

Removing the uniform flow is therefore an explicit modeling choice, not a
pressure gauge. P8.2 qualifies only the zero-friction, zero-mean-velocity
branch. The friction branch is manufactured-tested at the operator level but
is not registered as a public P8.2 application.

## Orthogonal lowering

The public compiler registry now has a third exact key. Lowering for this key
requires three periodic axes and assigns FFT bases to every Q, velocity,
pressure, molecular-field, stress, and force component. It does not reuse the
Plane wall parity, DCT/DST basis, distortion-stress odd-space rule, or
free-slip saddle solver.

The complete-stress constitutive calculation remains in the active-nematic
model package. The periodic Stokes implementation remains in
`linear_solvers`. Their concrete composition is owned by the periodic runtime
composition root, so the model layer does not acquire a reverse dependency on
the linear-solver layer.

The public boundary-policy objects remain field-level declarations. On a
`PeriodicBox`, `assign_boundaries(...)` lowers every face to the geometry's
periodic topology; `neumann_q()` and `free_slip_velocity()` do not create
fictitious walls. This keeps the public declaration uniform without treating
a wall-law name as the numerical periodic boundary condition.

## Fail-closed public application

The application runtime path is exactly `periodic_spectral`. The compiler
rejects unregistered pairs, runtime fallback, non-snapshot initialization,
unknown execution/workflow/invocation keys, discretization overrides,
nonzero-mean pressure gauges, frictional public runs, spectral refresh, and
unsupported execution modes before runtime construction.

The current P8.2 application requires:

- an external finite float64/float32 Q snapshot with five canonical
  components;
- projected semi-implicit Euler;
- the existing complete-stress Beris--Edwards constitutive kernels;
- a tensor-product spectral backend with all-periodic transforms;
- zero-mean pressure and zero-mean uniform velocity;
- runtime fallback disabled and TF32 disabled.

The application is package-owned and is reachable through the same public
`Simulation`, `compile_simulation(...)`, and `run_simulation(...)` entry
points as the Plane and Channel applications.

## Observation and exact restart

The periodic workflow writes canonical Q, velocity, and pressure snapshots,
finite diagnostics, metadata, checkpoints, and `COMPLETE` only after a
successful run. Diagnostics record spectral incompressibility, pressure mean,
and the norm of the three-component mean velocity.

Its checkpoint schema records spatial and spectral Q tensors, raw-byte
SHA-256 values, shape, dtype, runtime identity, backend restart identity, and
integrator progress. A two-step continuous CPU trajectory is byte-for-byte
identical in Q, u, and p to a one-step run followed by a one-step restart.
Tensor-byte tampering is rejected before advancement.

## Local verification

Manufactured periodic Stokes tests cover full-complex and Hermitian-half
storage, velocity recovery, pressure recovery, divergence, pressure gauge,
zero-mean uniform-mode removal, frictional uniform-mode retention, and
invalid-nullspace rejection.

The public application tests cover compiler/lowering identity, finite
trajectory execution, pressure and velocity means, continuous/restart byte
identity, unsupported-runtime rejection, and checkpoint tensor tampering.

Local results on 2026-09-25:

```text
P8.2 and architecture targeted suite: 105 passed
complete CPU suite: 2224 passed, 8 subtests passed
git diff --check: pass
```

No test failed, skipped, xfailed, or was deselected in this local environment.

## Remaining qualification

P8.2 is not closed until one frozen H100 qualification verifies:

1. clean commit/worktree and installed-package import identity;
2. CUDA-only backend identity and finite manufactured periodic Stokes smoke;
3. R128 and R320 periodic profiles with transform-call, timestep, peak
   allocated/reserved memory, graph-break, fallback, and state-hash evidence;
4. finite 100-step production trajectories;
5. continuous/restart identity and runtime/checkpoint tamper rejection;
6. immutable metadata, provenance, checksums, and final analyzer acceptance.

Until those gates pass, the capability catalog describes the periodic pair as
a local candidate, not a fully H100-qualified production combination. P8.3,
P8.4, Phase 9, default promotion, and benchmark claims are not authorized by
this local result.

## Compatibility debt recorded, not hidden

The periodic runtime still reaches the frozen v0.1 spectral `SpectralSolver`
and Euler integrator through two exact, tested compatibility edges. This is a
narrow migration debt at the runtime composition root. It is not a new
layer-wide permission and does not expose the legacy engine through the
public model, geometry, or boundary declarations.
