# Phase 8 P8.4.3: active-nematic Q conveniences and Plane application lowering

Status: `PASS_P8_4_3_Q_CONVENIENCES_AND_PLANE_APPLICATION_LOWERING`.

Baseline: `5904ec8c090bd7ecee9e7de014e0e29b5ff0d116` on
`next/pssolver-v0.2.0-architecture`, after P8.4.2 qualified the disconnected,
field-neutral Plane static-lifting plan and numerical operator. P8.4.3 adds
the active-nematic Q convenience declarations and lowers the first complete
Plane prescribed-Q application into that generic plan. It deliberately does
not connect the plan to a production runtime, workflow, output, checkpoint,
or restart path.

## Public Q policies

`pssolver.models.active_nematics.boundaries` owns three model-level
conveniences:

- `prescribed_q(face_values)` accepts an explicit constant symmetric,
  traceless Q tensor independently on each oriented wall face;
- `strong_homeotropic_q(scalar_order, face_normals)` constructs Q from the
  oriented unit wall normals;
- `strong_planar_q(scalar_order, face_directors, face_normals)` constructs Q
  from explicit unit directors tangent to their corresponding faces.

All three use the repository's single canonical convention,

```text
Q = (3 S / 2) (n n - I / 3),  S = lambda_max(Q),
```

whose identity is `de_gennes_S_lambda_max_v1`. They validate finite,
non-negative scalar order, three-dimensional unit normals/directors, planar
tangency, exact oriented-face agreement, symmetric-traceless explicit Q, and
spatially constant wall data. The result is the existing generic
`StaticPrescribedDirichletPolicy`; the model convenience layer does not own a
geometry, transform, lifting operator, runtime, or workflow.

The public capability catalog lists these policies as declarable but not yet
executable. This distinction is intentional: P8.4.3 makes declarations and
lowering machine-discoverable without claiming that a production timestep
can consume them.

## Qualified lowering contract

The first registered nonhomogeneous application is deliberately narrow:

- model field: the complete canonical five-component Q field;
- geometry: `PlaneSlab`;
- bounded axis: the existing single cell-centered Plane wall-normal axis;
- wall data: static constant Dirichlet values on both lower and upper faces
  for every Q component;
- velocity and pressure: the already-qualified Plane free-slip and pressure
  compatibility signatures.

The lowering plan embeds the content-addressed `StaticLiftingPlan`. For each
Q component it changes only the internal evolved representation from the
old even homogeneous basis

```text
FFT x FFT x DCT
```

to the homogeneous Dirichlet remainder basis

```text
FFT x FFT x DST.
```

The physical field contract remains

```text
Q_physical = Q_homogeneous + Q_lift.
```

The lifting plan binds the equation identity, physical-boundary identity,
geometry, grid, component ordering, independent lower/upper Q values, and
affine extension identity. Existing homogeneous Plane requests retain their
old DCT bases and their lowering metadata contains no lifting keys; the
optional extension therefore does not alter their canonical identity.

## Fail-closed boundary

P8.4.3 accepts only complete Plane Q coverage. It does not generalize
prescribed values to arbitrary fields, incomplete Q component sets,
PeriodicBox, Channel, time-dependent wall data, or spatially varying wall
data. Those requests retain the prior structured prescribed-boundary
rejection.

Even the qualified lifting plan is rejected by runtime binding with
`unsupported_lifting_runtime`. This is a deliberate stage boundary rather
than a missing safety check. P8.4.4 must connect materialization, one-time
linear lift correction, physical-state reconstruction, observations,
checkpoint identity, and restart before the public simulation can execute.

## Neumann scope

Only nonhomogeneous Dirichlet lifting is added here. Existing homogeneous
Neumann conditions remain supported through the existing DCT route and need
no lift. Prescribed nonzero Neumann flux is still unsupported and must be a
separate future capability with outward-normal sign conventions,
compatibility/nullspace rules, and geometry-specific validation. It must not
silently reuse the affine Dirichlet lifting contract.

## Verification and authorization

Local tests cover convention values, independent wall tensors, invalid Q and
orientation inputs, deterministic policy identity, exact Plane lift
coverage, DST remainder selection, unchanged homogeneous DCT lowering,
capability-catalog truthfulness, dependency boundaries, and the deliberate
runtime rejection. The complete local CPU suite is recorded in the matching
machine-readable result.

P8.4.3 changes no production default and uses no H100. P8.4.4 is eligible
for separate planning but is not implemented or authorized by this record.
Nonhomogeneous Neumann lifting, P8.4.5 H100 closure, P8.5 finite anchoring,
and Phase 9 remain outside this slice.
