# Phase 8 P8.4.2: field-neutral Plane static lifting

Status: `PASS_P8_4_2_PLANE_STATIC_LIFTING_AND_MANUFACTURED_ORACLE`.

Baseline: `9331f872cd24ae7999cbd8c4c60f8948b19e54e3` on
`next/pssolver-v0.2.0-architecture`, after P8.4.1 added generic prescribed
Dirichlet declarations.  P8.4.2 adds a disconnected numerical lifting plan,
materializer, and CPU manufactured oracle.  It does not connect lifting to a
public application, production runtime, workflow, checkpoint, or Q-specific
policy.

## Implemented mathematical contract

For every selected evolved component,

```text
phi_physical = phi_homogeneous + phi_lift,
phi_homogeneous = 0 on both prescribed Plane walls.
```

The first extension is static and affine along the unique wall-normal Plane
axis:

```text
phi_lift(s) = phi_lower + s (phi_upper - phi_lower),  0 <= s <= 1.
```

The Plane grid is cell-centered.  Materialized interior values therefore use
`s_j=(j+1/2)/N`; the mathematical wall contract is evaluated separately at
`s=0` and `s=1`.  The affine lift has an explicit zero Laplacian tensor, but
P8.4.2 does not equate that fact with a zero full linear correction.  A
model-supplied linear operator can be applied once during construction to an
isolated lift copy, producing an immutable, content-addressed
`L(phi_lift)` correction.  This correctly covers linear reaction, bulk, or
other model terms in addition to the Laplacian.

## Ownership

`pssolver.planning.lifting` owns tensor-free identities:

- `StaticLiftingComponentPlan`;
- `StaticLiftingPlan`;
- `StaticLiftExtension.AFFINE_WALL_NORMAL`;
The plan binds equation and boundary SHA-256 identities, geometry, shape,
lengths, axis names, cell-centered placement, wall-normal axis, component
order, independently prescribed lower/upper values, homogeneous-remainder
representation, and extension identity.  It imports no Torch runtime.

`pssolver.operators.lifting` owns numerical materialization:

- `PlaneStaticLiftingOperator`;
- `MaterializedLinearLiftCorrection`;
- `materialize_plane_static_lifting()`.

The operator materializes the component lifts once on the requested real
dtype and device.  It supplies physical reconstruction, inverse remainder
extraction, optional non-aliasing output workspaces, explicit affine
Laplacians, one-time model-supplied linear corrections, tensor SHA-256
identity, and mutation detection.  It imports no model, runtime, workflow, or
active-nematic package.

`pssolver.configuration.lifting` owns
`build_plane_static_lifting_plan()`.  This composition/lowering layer is the
only new code that reads equation-system, geometry, physical-boundary, and
planning contracts together.  Keeping that builder out of `planning`
preserves the Phase 0 dependency direction: planning objects do not import
equation systems.

## Fail-closed scope

This slice accepts only:

- `plane_slab` with exactly one bounded axis;
- cell-centered tensor-product domains;
- static constant prescribed Dirichlet values;
- physical evolved fields;
- both lower and upper wall faces;
- `float32` or `float64` materialization.

It rejects an unregistered geometry, multiple bounded axes, node-centered
placement, missing prescribed values, non-evolved fields, nonphysical
semantics, shape/dtype/device mismatches, nonfinite states or corrections,
unknown components, output/input aliasing, and materialized-lift mutation.

No existing lowering rejection is removed.  Public `Simulation` compilation
still stops at `unsupported_prescribed_boundary`; P8.4.3 must explicitly bind
the first Q-specific application before prescribed boundaries become
executable.

## Manufactured oracle

A model-neutral scalar field with unequal wall values is decomposed into its
affine lift and a sine homogeneous remainder on cell-centered grids.  A
second-order centered Dirichlet Laplacian applied to the recovered remainder,
plus the explicit affine-lift Laplacian, converges to the analytic physical
Laplacian:

| Wall-normal points | RMS error |
| ---: | ---: |
| 16 | 5.598158649073806e-3 |
| 32 | 1.4008891512969812e-3 |
| 64 | 3.503066853420464e-4 |

Observed orders are `1.9986095691699335` and `1.9996523773158625`.

## Neumann boundary scope

Existing homogeneous Neumann data, `partial_n phi = 0`, require no
nonhomogeneous lift and remain on the existing DCT path.  P8.4.2 does not
support prescribed nonzero flux, `partial_n phi = g`.  Such support requires
a separately designed nonhomogeneous-Neumann lifting or tau contract,
including outward-normal sign, compatibility/nullspace conditions, and
geometry-specific validation.  It must not silently reuse this Dirichlet
affine lift.  Finite Robin/surface-energy anchoring remains P8.5.

## Authorization boundary

P8.4.2 is local CPU operator work and uses no H100.  It makes P8.4.3 eligible
for separate planning.  Q conveniences, application lowering, runtime
connection, checkpoint changes, H100 qualification, nonhomogeneous Neumann,
P8.5, and Phase 9 remain unauthorized.
