# Phase 8 P8.5.0 planning: field-neutral Robin laws and finite Q anchoring

Status: `PASS_P8_5_0_PLANNING_IMPLEMENTATION_NOT_AUTHORIZED`.

Baseline: `104258d5787921e527c61b180b6eb828013d184b` on
`next/pssolver-v0.2.0-architecture`, after the authoritative P8.4.5 H100
closure.  P8.4.5 makes P8.5 planning eligible.  This record freezes the
physical law, ownership, first numerical pilot, rejection semantics, and gate
order.  It implements no Robin declaration, tau row, bounded-axis operator,
runtime path, compiler registration, or production-default change.

## Decision

The generic public boundary law is a field-neutral Robin condition on one
logical scalar component and one oriented face:

```text
alpha phi + beta (n dot grad phi) = gamma.
```

`n` is always the outward unit normal supplied by the geometry.  Consequently
the normal derivative is `-partial_z` on the lower face of a Plane slab and
`+partial_z` on the upper face.  The declaration stores static finite
coefficient/value identities and physical semantics; it does not name FFT,
DCT, DST, a Robin eigenbasis, tau rows, lifting, a model, or a solver.

The first implementation will admit constant real `alpha`, `beta`, and
`gamma` independently on each bounded face and component.  It will preserve
the raw coefficient triple in metadata.  Multiplying all three coefficients
by a common nonzero factor describes an equivalent continuum law but does not
silently collapse provenance identities.  Any normalization used internally
by an operator is separate, deterministic lowering metadata.

Finite Q anchoring is a model specialization over this generic law, not a
generic boundary-core feature.  It is also not strong Dirichlet anchoring with
a loose numerical tolerance.

## Variational finite-anchoring law

The first Q pilot uses the one-constant elastic and quadratic surface
energies

```text
F_bulk = (K_Q / 2) integral_V (partial_k Q_ij)(partial_k Q_ij) dV,
F_surface = (W / 2) integral_boundary (Q_ij - Q*_ij)(Q_ij - Q*_ij) dA.
```

Variation in the symmetric-traceless tensor space gives the natural wall
condition

```text
K_Q n_k partial_k Q_ij + W (Q_ij - Q*_ij) = 0,
```

or, componentwise after a metric-consistent projection,

```text
W Q_component + K_Q (n dot grad Q_component) = W Q*_component.
```

Thus the generic Robin coefficients are `alpha=W`, `beta=K_Q`, and
`gamma=W Q*`.  `K_Q` means the actual coefficient multiplying the qualified
one-constant Q-gradient energy in the model declaration.  The adapter must
not guess that an ambiguously named `K`, `L1`, or Frank constant has the same
normalization.

The extrapolation length is

```text
ell_a = K_Q / W
```

for `W > 0`.  `W = 0` reduces to homogeneous Neumann and retains its existing
null-space/gauge implications.  The formal `W -> infinity` limit approaches
strong anchoring, but infinity is never a runtime value and the qualified
strong-Dirichlet path remains a distinct policy and implementation.

The five stored independent Q components are coordinates on the
symmetric-traceless tensor space, not five unrelated scalar order parameters.
The Q adapter must derive from full tensor contractions, verify that the
induced five-component metric cancels consistently between the one-constant
bulk and quadratic surface variations, and require a symmetric-traceless
target Q.  Only then may it emit componentwise Robin data.

## Ownership

### Generic physical declaration

`pssolver.core.boundary` owns a tensor-free `RobinBoundaryCondition`-level
identity: boundary family, static coefficient providers, component, oriented
face, semantic, units/dimensional role, and canonical metadata.  It contains
no tensor, callable, geometry object, transform, tau row, or model object.

`pssolver.boundaries` owns a field-neutral policy that maps registered evolved
field components and oriented faces to those Robin laws.  It validates
coverage, uniqueness, finite constants, and periodic-face rejection.  It must
work for a future scalar, vector, or tensor field without importing active
nematics.

### Planning and geometry

`pssolver.planning` owns a `RobinRequirement` and immutable lowering plan.
The geometry owns oriented faces, outward normals, wall-normal axis, grid
placement, and metrics.  Planning checks the registered field role and
component coverage before any allocation, then selects an explicitly
qualified bounded-axis backend.  Unsupported combinations fail with a
structured capability error; no Dirichlet or Neumann fallback is allowed.

### Numerical operator

`pssolver.operators` owns the discrete boundary residual and bounded-axis
Robin/tau operator.  Periodic directions remain Fourier spectral.  A generic
DCT or DST does not satisfy arbitrary finite Robin data, so the first pilot
must not compute a full DCT/DST solution and patch the wall afterward.

P8.5.2 will compare and freeze one exact spectral bounded-axis construction:

1. a Robin eigenbasis with a qualified root/eigenvalue plan; or
2. a Chebyshev/ultraspherical tau or bordered operator with explicit wall
   equations.

The method decision is an operator ADR backed by manufactured convergence,
conditioning, memory, and batched-GPU feasibility.  `tau` is an implementation
choice below the public Robin law, never the name of the physical boundary
condition.  No runtime work begins until that decision closes.

### Active-nematic specialization

`pssolver.models.active_nematics.boundaries` owns the quadratic finite-Q
anchoring convenience.  It validates `W >= 0`, the explicit qualified
`K_Q > 0`, Q convention, target symmetric-traceless structure, component
ordering, and face-normal use.  Planar and homeotropic target constructors
reuse the canonical convention

```text
Q* = (3 S / 2) (n n - I / 3).
```

The convenience layer returns generic Robin data.  It does not construct a
geometry, transform plan, operator, runtime, or checkpoint.

## First qualified slice

The first scientific slice is deliberately narrow:

```text
complete_stress_beris_edwards
  + plane_slab
  + free_slip_velocity
  + one_constant_q_elasticity
  + static quadratic finite Q anchoring on both z faces
  + constant W and constant target Q per face
  + legacy_production control path
```

Lower and upper faces may have different finite `W` and target Q values, but
each face is spatially and temporally constant.  The first operator is
float64-first and must have an independent CPU manufactured oracle before a
GPU path is authorized.

## Manufactured and limiting oracles

The scalar operator oracle uses a smooth known `phi(z)` and derives both the
bulk forcing and the two oriented-face values of

```text
r_wall = alpha phi + beta (n dot grad phi) - gamma.
```

It separately reports interior residual, lower-wall residual, upper-wall
residual, solution error, and observed convergence.  Lower/upper signs are
tested independently.  A passing implementation requires at least
second-order convergence for the first discretization and near-roundoff wall
residual for functions represented exactly by the chosen basis.

Additional limiting controls compare:

- `W = 0` with the existing homogeneous-Neumann physical law, including its
  null-space treatment;
- increasing finite `W` with convergence toward the already-qualified strong
  Dirichlet solution without ever passing infinity;
- zero target Q and nonzero target Q;
- equal and unequal lower/upper face data.

The Q oracle checks the full tensor variational residual and the lowered five
component residual, not only the saved wall value.

## State, restart, and provenance

Checkpoint identity must include raw Robin coefficient identities, any
operator-normalized coefficients, outward-normal convention, surface-energy
law identity, Q convention, target-Q identity, bounded-axis operator kind,
operator-plan identity, geometry, dtype/device intent, runtime, and backend.
Operator plans and cached factorizations are rebuilt and hash-verified before
target mutation.  Continuous and split/restarted trajectories must be
byte-for-byte identical within one runtime.

Negative gates cover coefficient, face orientation, target Q, Q convention,
surface-law, operator-kind, eigenvalue/tau-plan, geometry, shape, dtype,
runtime/backend, tensor payload, and file-record tampering.  A rejected
restart must not mutate the target.

## Performance contract

There is no registry lookup, policy dispatch, coefficient-provider call,
root solve, matrix factorization, or new plan allocation in the timestep hot
loop.  Robin eigenvalues or tau/bordered factorizations are constructed once
and cached by exact device, dtype, shape, geometry, and coefficient identity.
H100 closure must report timestep, peak allocated/active/reserved memory,
transform and bounded-axis solve counts, graph breaks, fallback state,
conditioning diagnostics, numerical equivalence, and exact restart.

## Planned slices

1. **P8.5.0 — planning (this record).** Freeze the physical law, ownership,
   first scope, operator-decision boundary, and gates.  No runtime code.
2. **P8.5.1 — field-neutral Robin declarations.** Add static coefficient
   identities, public per-field/per-face policy, metadata, stable hashing, and
   structured rejection.  No Q-specific names in the generic layer.
3. **P8.5.2 — bounded-axis method ADR and CPU operator.** Select Robin
   eigenbasis or tau/bordered construction, implement the scalar Plane
   operator, and pass oriented-wall manufactured and limiting oracles.
4. **P8.5.3 — generic Plane lowering and runtime state.** Bind a registered
   scalar evolved-field pilot, operator-plan/cache identity, observation, and
   checkpoint semantics without active-nematic coupling.
5. **P8.5.4 — finite-Q anchoring specialization.** Add the one-constant
   quadratic surface-energy adapter, five-component metric checks, target-Q
   conveniences, and only the first qualified application combination.
6. **P8.5.5 — workflow, restart, and CPU closure.** Qualify continuous/split
   identity, tamper rejection, installed-package behavior, limits, and
   unchanged P8.4 controls.
7. **P8.5.6 — H100 non-regression closure.** Qualify performance, live memory,
   transform/solve counts, no fallback, conditioning, provenance, checksums,
   and numerical equivalence before P8.6 planning.

Each implementation slice requires separate user authorization.  Completing
this planning record makes P8.5.1 eligible; it does not implement or authorize
the complete P8.5 runtime.

## Explicit non-goals

The first P8.5 pilot does not add prescribed nonzero Neumann flux,
time-dependent or trainable Robin data, spatially varying `W` or target Q,
nonlinear or degenerate-planar surface energies, multiple elastic constants,
curved surfaces, Channel finite anchoring, arbitrary Python callables,
automatic fallback, compiled-runtime promotion, a production-default change,
optimal control, Phase 9, or a paper-benchmark claim.

In particular, the degenerate planar Fournier--Galatola-type surface energy
is not silently approximated by a single target-Q quadratic potential.  It
requires its own model law and qualification.

## Authorization boundary

P8.5.0 planning is complete and P8.5.1 declaration work is eligible for the
next explicit request.  P8.5.1 implementation, numerical-method selection,
runtime work, H100 submission, nonhomogeneous Neumann work, P8.6, Phase 9,
production-default changes, and compiled-runtime promotion remain
unauthorized.
