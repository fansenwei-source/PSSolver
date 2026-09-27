# Phase 8 P8.5.4: quadratic finite-Q anchoring specialization

Status: `PASS_P8_5_4_QUADRATIC_FINITE_Q_ANCHORING_SPECIALIZATION`.

Baseline: `fb4b745ffe4289d83c39a7304a7a82ae02c58eca`, the P8.5.3
recorded closure. Implementation commit:
`ce9d8ec4d68515663e886298b2241d8817d69b64`.

P8.5.4 adds the model-owned specialization that converts the qualified
one-constant quadratic Q surface energy into five generic Robin laws. It
also adds a fail-closed internal composition/lowering pilot for the first
Plane complete-stress application. It does not connect a complete Q
timestep, workflow, public runner, production runtime, or GPU path.

## Physical law and coefficient ownership

The qualified energies are

```text
F_bulk    = (K_Q/2) integral_V (partial_k Q_ij)(partial_k Q_ij) dV,
F_surface = (W/2)   integral_A (Q_ij-Q*_ij)(Q_ij-Q*_ij) dA.
```

Variation in the symmetric-traceless tensor space gives

```text
K_Q (n dot grad Q_ij) + W (Q_ij-Q*_ij) = 0.
```

The emitted generic Robin coefficients for every independent Q component
are therefore

```text
alpha = W,
beta  = K_Q,
gamma = W Q*_component.
```

`K_Q` is never inferred from a Frank constant or from an ambiguously named
`K` or `L1`. The first application requires it to match exactly every
qualified `ldg_l1` occurrence in the complete-stress equation declaration.
`K_Q` must be positive and finite; `W` must be finite and non-negative.

For `W>0`, the metadata reports the extrapolation length `K_Q/W`. `W=0`
emits homogeneous Neumann laws. Even though different targets then yield the
same generic Robin coefficients, the model specialization retains distinct
target-Q and surface-law provenance. Strong Dirichlet anchoring remains a
separate policy; infinity is not an accepted coefficient.

## Five-component metric oracle

The stored component order is

```text
(Qxx, Qxy, Qxz, Qyy, Qyz),  Qzz=-(Qxx+Qyy).
```

The full tensor contraction induces the Gram matrix

```text
G = [[2,0,0,1,0],
     [0,2,0,0,0],
     [0,0,2,0,0],
     [1,0,0,2,0],
     [0,0,0,0,2]].
```

Both the one-constant bulk variation and the quadratic surface variation
contain this same positive-definite matrix. The CPU oracle constructs the
five full-tensor coordinate basis matrices, derives `G` by tensor
contraction, and verifies

```text
projected full-tensor variation
    = G [K_Q normal_derivative(q) + W(q-q*)].
```

Because `G` is invertible, the full symmetric-traceless variational law is
equivalent to the five componentwise Robin laws. The implementation does not
treat the five values as five independent scalar order parameters.

## Target-Q conveniences

`QuadraticFiniteQAnchoring` owns the surface-law identity, explicit `K_Q`,
per-face `W`, canonical target Q, Q convention, component metric, and the
derived generic Robin policy. Explicit targets must be constant, finite,
symmetric, and traceless.

The homeotropic and explicit-planar conveniences use the sole canonical
convention

```text
Q* = (3S/2)(nn-I/3),  S=lambda_max(Q).
```

Homeotropic targets use the oriented face normal as director. Planar targets
require an explicit unit tangent director and reject a director with a normal
component. Lower and upper walls may use different strengths and targets.

## First application lowering

The internal P8.5.4 application pilot accepts only:

- `complete_stress_beris_edwards`;
- `plane_slab` with the already-qualified free-slip velocity and Neumann
  pressure compatibility laws;
- `legacy_production` as the declared control runtime;
- exactly the lower and upper faces of wall-normal axis 2;
- float64/cell-centered generic scalar lowering inherited from P8.5.3;
- an explicit anchoring `K_Q` exactly equal to every model `ldg_l1`.

It replaces only the Q boundary assignment, retains the qualified
hydrodynamic assignments, records the anchoring metadata and identity, and
lowers the five ordered Q components into five P8.5.3 scalar Robin plans.
The component plans retain the source simulation, boundary, raw coefficient,
operator, and normal-orientation identities.

The pilot entry points stay inside
`pssolver.configuration.finite_q_anchoring`; top-level `pssolver`, the frozen
configuration facade, public `compile_simulation()`, and public
`run_simulation()` are unchanged. Production lowering continues to reject
Robin Q boundaries with `unsupported_robin_boundary`.

## Scope boundary

P8.5.4 does not implement:

- a five-component runtime or complete Beris--Edwards timestep;
- workflow output, file-backed checkpoint, or restart;
- GPU materialization or H100 qualification;
- time-dependent, spatially varying, or trainable surface data;
- multiple elastic constants or nonlinear/degenerate-planar surface energy;
- prescribed nonzero Neumann flux;
- Channel or curved-surface finite anchoring;
- public executable Robin combinations;
- a production-default change or scientific benchmark claim.

P8.5.5 may now bind the five component plans into workflow and restart
semantics and perform the full CPU closure. That work is eligible but is not
implemented by this record.

## Verification

P8.5.4 implementation tests before the record: `14 passed, 1 deselected`.
Related Q/Robin/architecture tests: `80 passed, 1 deselected`.

The full local suite passed with `2373 passed, 8 subtests passed`, with zero
failures, skips, xfails, or deselections.

The archive source list includes the two P8.5.4 records. The user-owned
untracked PDF was not regenerated or changed.
