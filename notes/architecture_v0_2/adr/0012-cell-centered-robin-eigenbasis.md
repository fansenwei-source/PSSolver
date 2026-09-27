# ADR 0012: use a cell-centered Robin eigenbasis for the first finite-anchoring slice

Status: accepted for the P8.5.2 CPU oracle; production runtime connection is
not authorized by this decision.

## Context

P8.5.1 introduced the field-neutral physical law

```text
alpha phi + beta (n dot grad phi) = gamma
```

without assigning a numerical method.  Arbitrary finite Robin data are not a
DCT or DST parity condition.  P8.5.2 must select an exact bounded-axis
spectral construction while preserving the already-qualified uniform,
cell-centered Plane grid, the periodic Fourier axes, and the existing field
storage layout.

Two candidates were frozen by P8.5.0:

1. a coefficient-specific Robin Sturm--Liouville eigenbasis; and
2. a Chebyshev/ultraspherical tau or bordered operator.

## Decision

The first qualified method is a coefficient-specific continuum Robin
eigenbasis, sampled at the existing cell centers.  Nonhomogeneous static data
are removed by a construction-time affine lift.  The homogeneous remainder is
expanded in eigenfunctions of `-d2/dz2` satisfying both oriented Robin laws.

For lower and upper coefficients with `beta > 0`, define

```text
h0 = alpha_lower / beta_lower
hL = alpha_upper / beta_upper.
```

With outward normals `n_lower=-z_hat` and `n_upper=+z_hat`, the homogeneous
conditions are

```text
phi'(0) = h0 phi(0),
phi'(L) = -hL phi(L).
```

For nonzero wave number `k`, a lower-wall-compatible eigenfunction is

```text
v_k(z) = cos(k z) + (h0/k) sin(k z).
```

The dimensionless roots `x=kL` solve

```text
(a b - x^2) sin(x) + x(a+b) cos(x) = 0,
a=h0 L,
b=hL L.
```

For the first physical slice, `alpha >= 0` and `beta > 0` on both faces.
Except for pure Neumann data, one root is bracketed in every interval
`(m pi, (m+1) pi)`.  Pure homogeneous Neumann data use the exact roots
`m pi`, including the constant zero mode.  Roots are found once by
deterministic bracketed bisection and are part of the immutable plan identity.

The affine lift `ell(z)=c0+c1 z` exactly satisfies both nonhomogeneous face
laws.  The operator solves

```text
(mass - d2/dz2) phi = f
```

by transforming `phi-ell`, dividing by `mass+k^2`, and reconstructing the
physical field.  A zero-mass pure-Neumann solve is rejected because its
constant null mode is unresolved.

## Why not Chebyshev/tau in the first slice

A Chebyshev or ultraspherical tau formulation remains a valid future backend,
especially for spatially varying coefficients, nonlinear wall laws, or more
general bounded operators.  It is not selected first because it would either
change the current uniform cell-centered grid or introduce interpolation and
a second state representation.  That would mix a boundary-method experiment
with geometry, state-layout, observation, restart, and memory migrations.

The selected eigenbasis instead preserves the current grid and periodic FFT
axes, diagonalizes the constant-coefficient bounded Laplacian, and admits a
batched matrix implementation on a future GPU path.

## Deliberate cost and backend boundary

The reference materialization stores a sampled basis and its inverse.  The
bounded transform is therefore dense `O(N_z^2)`, not an `O(N_z log N_z)` DCT
or DST.  This is acceptable for the first `N_z=80/160`-class pilot and makes
the mathematical oracle explicit, but it is not claimed to be the final fast
implementation.  The public boundary law names Robin physics, not this
backend; a future pruned transform, structured recurrence, or tau backend may
replace it behind the same plan boundary after independent qualification.

Root solves, matrix construction, inversion, conditioning checks, and affine
lift construction occur once.  They are forbidden from the eventual timestep
hot loop.

## First qualified numerical scope

- one bounded Plane axis;
- uniform cell-centered samples;
- real finite constant coefficients;
- `alpha >= 0`, `beta > 0` on both faces;
- float64 CPU oracle;
- static affine nonhomogeneous lift;
- homogeneous pure-Neumann limit with explicit null-mode rejection;
- independent lower and upper laws.

Prescribed nonzero Neumann flux, negative impedance, time- or space-dependent
coefficients, a GPU runtime, Q-specific anchoring, Channel, compiled-runtime
promotion, and a production-default change remain outside P8.5.2.

## Consequences and gates

The CPU oracle must demonstrate:

- deterministic plan and coefficient identity;
- correct lower/upper outward-normal signs;
- round-trip and exactly represented wall residuals near float64 roundoff;
- at least nominal second-order convergence for a smooth manufactured
  Helmholtz problem;
- the homogeneous-Neumann cosine limit and null-mode rejection;
- convergence toward the existing strong-Dirichlet control as finite
  impedance increases;
- bounded sampled-basis conditioning for the qualified cases;
- no runtime connection or H100 claim.

P8.5.3 may lower this plan for a generic registered scalar Plane field only
after a separate explicit request.
