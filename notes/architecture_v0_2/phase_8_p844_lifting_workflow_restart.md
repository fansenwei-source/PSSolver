# Phase 8 P8.4.4: Plane static-lifting runtime, workflow, and restart

Status: `PASS_P8_4_4_PLANE_STATIC_LIFTING_WORKFLOW_RESTART`.

Baseline: `fd2bd03f235711d57eff098ee0d90f7bb7213f58` on
`next/pssolver-v0.2.0-architecture`. P8.4.3 already lowered complete Plane
prescribed-Q boundaries into the field-neutral `StaticLiftingPlan`; P8.4.4
connects that plan to the qualified `legacy_production` runtime, physical
observations, checkpoint identity, and exact restart. It does not authorize
an H100 run or broaden the boundary-condition scope.

## Representation and ownership

The evolved `Fields` authority remains the homogeneous remainder

```text
Q_homogeneous = Q_physical - Q_lift,
```

in the native `FFT x FFT x DST` basis. The new field-neutral
`PlaneStaticLiftingRuntime` owns the immutable, content-addressed lift and one
preallocated physical workspace. Constitutive kernels and Stokes stresses see

```text
Q_physical = Q_homogeneous + Q_lift,
```

while the integrator, spectral refresh, and checkpoint tensors continue to
operate on the homogeneous remainder. There is no per-step registry lookup or
second persistent copy of the lift: the batched lift used by reconstruction
is a view of the operator-owned, identity-checked storage.

For the affine Plane extension, the wall-normal gradient correction is the
constant difference of the two wall values divided by the physical height,
and the lift Laplacian is explicitly zero. The model-supplied linear lift
term is materialized and transformed once during construction. Bulk,
advection, alignment, molecular-field, active-stress, reactive-stress, and
distortion-stress evaluations all consume the reconstructed physical Q or its
corrected derivatives.

## Qualified runtime boundary

The connected combination is deliberately narrow:

- equation system: complete-stress Beris--Edwards;
- geometry: `PlaneSlab`;
- prescribed field: all five canonical Q components;
- boundary data: static, constant, two-sided Dirichlet values;
- velocity: the existing qualified free-slip policy;
- pressure: the existing qualified Neumann compatibility policy;
- runtime: `legacy_production` only.

The compiled Plane path still fails closed with
`unsupported_lifting_runtime`. Existing homogeneous Plane, periodic, and
Channel paths keep their previous representation and defaults.

## Workflow and output contract

Public `compile_simulation()` retains the exact prescribed boundary
declaration and its lowering identity. `run_simulation()` forwards that
application declaration to the package-owned Plane construction path.

Saved `Q_*.npy` observations are physical Q. Velocity and pressure output are
unchanged. Initial-condition metadata distinguishes the physical projected-Q
identity from the evolved homogeneous-remainder identity. Runtime metadata
records the lifting plan, materialized lift, linear correction, canonical Q
convention, and the physical/evolved representation split.

## Checkpoint and restart contract

Checkpoint tensors are the evolved spatial and spectral homogeneous
remainders. The checkpoint metadata additionally binds:

- lifting-plan SHA-256;
- materialized-lift SHA-256;
- component order, shape, dtype, and device;
- every materialized linear-correction identity;
- the canonical Q-convention identity;
- the explicit representation names.

Restore compares the complete lifting identity and verifies the live
materialized lift before mutating target tensors. It then validates every
checkpoint tensor shape and dtype before the first copy. A tampered lifting
identity is therefore rejected without partial target mutation. A split run
and a continuous run produce byte-identical Q-remainder, physical Q, velocity,
and pressure tensors in the local CPU oracle.

## Neumann scope

P8.4.4 does not implement prescribed nonzero Neumann flux. Existing
homogeneous Neumann conditions continue to use their established DCT path and
need no lift. Nonhomogeneous Neumann requires a separate future contract for
outward-normal signs, compatibility and nullspace rules, flux provenance,
and geometry-specific validation; it must not be inferred from this
Dirichlet implementation.

## Verification and authorization

Local verification covers narrow runtime binding, compiled-path rejection,
DST remainder storage, physical observations, convention and boundary
identity, finite evolution, exact continuous/split restart, fail-closed
lifting-identity tamper rejection, and public-runner output reconstruction.
The complete CPU regression suite passes with 2290 tests and 8 subtests.

P8.4.4 changes no default runtime and uses no H100. P8.4.5 H100 closure is
eligible for separate authorization but is not performed by this record.
Finite/Robin anchoring, prescribed nonzero Neumann flux, Channel strong
anchoring, compiled-runtime lifting, Phase 9, and benchmark claims remain
outside this slice.
