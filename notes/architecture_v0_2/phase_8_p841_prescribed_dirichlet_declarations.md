# Phase 8 P8.4.1: generic static prescribed-Dirichlet declarations

Status: `PASS_P8_4_1_GENERIC_PRESCRIBED_DIRICHLET_DECLARATIONS`.

Baseline: `64f44087d808686f895f079b1d14966d39345bce` on
`next/pssolver-v0.2.0-architecture`.  This slice implements only the generic
declaration and composition layer authorized after the P8.4.0 plan.  It does
not implement a numerical lift, change an existing runtime, or make a new
model/geometry/boundary combination executable.

## Result

P8.4.1 introduces a field-neutral representation of static scalar Dirichlet
data.  The representation has deterministic JSON metadata and SHA-256
identity, rejects Boolean and non-finite values, and contains no tensor,
callable, transform, solver, geometry, or model object.

`pssolver.boundaries.prescribed_dirichlet()` accepts values keyed by logical
component, axis, and oriented face.  `assign_boundaries()` can now combine
that policy with existing homogeneous policies.  Composition validates the
actual equation-field registry and geometry:

- prescribed data may target any evolved logical field;
- every component on every bounded face must be covered exactly once;
- lower and upper face values are independent;
- periodic faces are supplied by topology and cannot carry prescribed data;
- algebraic, transient, and diagnostic fields cannot receive this policy;
- unknown components, axes, duplicate entries, missing entries, and extra
  entries fail during tensor-free composition.

The implementation is intentionally not Q-specific.  Neither the core value
contract nor the public prescribed policy imports or names active nematics,
directors, scalar order, Torch, NumPy, a runtime, or a transform backend.
Strong-Q convenience policies remain P8.4.3 work.

## Fail-closed execution boundary

Before P8.4.1, lowering selected a transform solely from `BoundaryKind`.
Allowing a nonzero Dirichlet declaration without another guard would therefore
silently select the existing homogeneous DST path and lose the prescribed
value.  P8.4.1 adds the structured rejection
`unsupported_prescribed_boundary`.  Any current lowering attempt containing
nonhomogeneous prescribed data stops before basis/runtime construction with
the affected component and faces in canonical context metadata.

This rejection is part of the implementation, not a temporary test
workaround.  It remains authoritative until P8.4.2 supplies and independently
qualifies a numerical lifting plan.

## Public contracts added

- `StaticConstantBoundaryValue`
- `PrescribedDirichletBC`
- `PrescribedFaceValue`
- `StaticPrescribedDirichletPolicy`
- `prescribed_dirichlet()`
- the widened `BoundaryPolicy` accepted by `assign_boundaries()`
- `LoweringRejectionCode.UNSUPPORTED_PRESCRIBED_BOUNDARY`

The base `BoundaryCondition(..., is_homogeneous=False)` remains invalid.  A
caller must use the typed prescribed condition so nonhomogeneous data cannot
exist without content identity.

## Verification and scope

Tests cover canonical identity, finite-value validation, deterministic policy
ordering, a generic scalar transport field named `concentration`, independent
wall values, topology/coverage/role failures, public exports, dependency
boundaries, historical-plan consistency, and structured lowering rejection.
The existing homogeneous API remains unchanged.

No H100 work is needed for a tensor-free declaration slice.  No lifting
operator, Q convenience, checkpoint format, observation reconstruction,
production default, runtime promotion, P8.5 feature, or Phase 9 feature is
implemented or authorized here.

P8.4.1 makes P8.4.2 eligible for separate planning.  P8.4.2 implementation
and any H100 submission remain separately unauthorized.
