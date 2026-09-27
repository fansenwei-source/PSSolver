# Phase 8 P8.5.2: cell-centered Robin eigenbasis CPU oracle

Status: `PASS_P8_5_2_CELL_CENTERED_ROBIN_EIGENBASIS_CPU_ORACLE`.

Baseline: `d98fcf286979ae47c36388beac84b9e68f97b121`, the P8.5.1 declaration
closure.  Implementation commit:
`d75d4db2195e5512f9034427190f68f9ac7929de`.

P8.5.2 closes the numerical-method decision and implements an independent
float64 CPU oracle.  It does not connect Robin boundaries to public
simulation lowering, a runtime, a checkpoint, active-nematic Q, or a GPU.

## Method decision

ADR 0012 selects a coefficient-specific continuum Robin eigenbasis sampled
on the existing uniform cell-centered Plane grid.  A static affine lift
removes the nonhomogeneous right-hand side, and the homogeneous remainder is
expanded in eigenfunctions of the wall-normal Laplacian.

For

```text
alpha_lower phi(0) - beta_lower phi'(0) = gamma_lower,
alpha_upper phi(L) + beta_upper phi'(L) = gamma_upper,
```

the first qualified slice requires `alpha >= 0` and `beta > 0` on both
faces.  With `h0=alpha_lower/beta_lower` and
`hL=alpha_upper/beta_upper`, its nonzero modes are

```text
v_k(z) = cos(kz) + (h0/k) sin(kz)
```

and `x=kL` is a root of

```text
(a b - x^2) sin(x) + x(a+b) cos(x) = 0,
a=h0 L,
b=hL L.
```

The plan finds these roots once using deterministic bracketed bisection.
Pure homogeneous Neumann data use exact cosine roots, including the constant
zero mode.  A zero-mass solve then rejects the unresolved constant null mode.

Chebyshev/ultraspherical tau remains a possible later backend but is not the
first implementation.  Introducing it now would change the qualified
cell-centered grid or require interpolation and a second state
representation.  The selected method preserves the current grid, periodic
FFT axes, and field storage layout.

## Operator and ownership

`pssolver.planning.robin` owns the tensor-free coefficient-specific root
plan and its canonical identity.  `pssolver.operators.robin` owns the
materialized sampled basis, inverse basis, affine lift, Helmholtz
apply/solve, and continuous oriented-wall residual.

The materialized reference is deliberately dense.  It stores the sampled
basis and its inverse and performs `O(N_z^2)` bounded transforms.  For the
anticipated `N_z=80` and `N_z=160` pilot sizes, the two float64 matrices use
about 100 KiB and 400 KiB respectively before small vectors.  This is a
clear, batched-GPU-feasible oracle, not a claim that the final Robin backend
is an `O(N log N)` fast transform.

Root solving, matrix construction, inversion, conditioning checks, and lift
construction occur only during materialization.  No registry lookup, root
solve, or factorization is permitted in a future timestep.

## CPU evidence

The asymmetric manufactured Helmholtz problem uses distinct lower/upper
coefficients and nonzero static data.  Relative solution errors for
`N_z=8,16,32` are

```text
3.113397128132618e-4
7.960597550283833e-5
2.0200222680378565e-5
```

which gives observed orders greater than 1.9.  Interior apply/solve
residuals are below `2e-12`; lower and upper oriented-wall residuals are
below `2e-13` and `2e-12` respectively.  Exactly represented modal data
round-trip to float64 precision and satisfy both continuous wall equations
near roundoff.

The pure-Neumann control reproduces a cosine mode to float64 roundoff and
rejects the zero-mass null mode.  Increasing equal finite impedances from
`10` to `100` to `1000` reduces the error relative to the qualified strong
Dirichlet control from approximately `1.47e-1` to `1.60e-2` to `1.61e-3`.
Raw common rescaling leaves normalized eigenvalues unchanged while retaining
distinct declaration and plan provenance.

The sampled-basis condition numbers in the qualified manufactured and
limiting controls remain close to one and below the explicit `1e10`
fail-closed ceiling.

## Scope boundary

P8.5.2 does not implement:

- public Robin simulation lowering;
- a generic Plane runtime or runtime state;
- finite-Q surface-energy anchoring;
- prescribed nonzero Neumann flux;
- negative impedance or zero `beta`;
- spatially or temporally varying coefficients;
- Channel or curved-surface Robin data;
- checkpoint/restart extensions;
- a GPU kernel or H100 qualification;
- a production-default change or paper benchmark.

The existing structured `unsupported_robin_boundary` rejection therefore
remains active.  P8.5.3 generic scalar Plane lowering and runtime work is now
eligible for a separate explicit request; it is not implemented here.

## Verification

Targeted P8.5 tests: `32 passed`.

Full local suite: `2347 passed, 8 subtests passed`, with zero failures,
skips, xfails, or deselections.

The architecture archive source list now includes ADR 0012 and the two
P8.5.2 records.  The user-owned untracked PDF was not regenerated or changed.
