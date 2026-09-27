# Phase 8 P8.5.1: field-neutral static Robin declarations

Status: `PASS_P8_5_1_FIELD_NEUTRAL_STATIC_ROBIN_DECLARATIONS`.

Baseline: `36f49a76f80110e48dd3cdb96f3ef94bb988e5c0`, the P8.5.0 planning
record on `next/pssolver-v0.2.0-architecture`.  Implementation commit:
`34c4c8f1884aefe8f700ad387a2be324a6a3e078`.

This slice implements only the tensor-free declaration, composition,
identity, discovery, and fail-closed lowering boundary authorized by P8.5.0.
It does not implement a Robin eigenbasis, tau/bordered operator, finite-Q
surface energy, runtime path, checkpoint extension, or executable
model/geometry/boundary combination.

## Generic law and identity

The canonical public law is

```text
alpha phi + beta (n dot grad phi) = gamma,
```

where `n` is the outward unit normal.  `StaticRobinCoefficients` records the
raw finite real `alpha`, `beta`, and `gamma` values, their dimensional roles,
the outward-normal convention, and the fact that the stored triple is not
operator-normalized.  `alpha=beta=0` is rejected because it does not define a
boundary law.  Boolean and non-finite coefficients are rejected.

Raw coefficient triples have deterministic canonical-JSON SHA-256 identity.
Two triples related by a common nonzero scale represent the same continuum
equation but intentionally retain different provenance identities.  A future
operator may add separately identified normalized coefficients; declaration
construction does not perform that normalization.

`StaticRobinBC` is the only accepted low-level Robin condition.  Constructing
the base `BoundaryCondition(kind=ROBIN)` is rejected so a Robin law cannot
exist without coefficient identity.  `gamma=0` is recorded as a homogeneous
Robin law; a nonzero `gamma` is nonhomogeneous without losing its typed data.

## Field-neutral public policy

`pssolver.boundaries.robin()` constructs a `StaticRobinBoundaryPolicy` from
per-component, per-axis, per-oriented-face `RobinFaceLaw` objects or
`(alpha, beta, gamma)` tuples.  The policy:

- can target any registered evolved logical field;
- is independent of scalar, vector, or tensor model meaning;
- requires exact coverage of every component on every bounded face;
- permits independent lower and upper coefficient triples;
- fills periodic faces from geometry and rejects user Robin data there;
- rejects algebraic, transient, diagnostic, unknown, duplicate, missing, and
  extra targets during tensor-free composition;
- contains no Torch tensor, NumPy array, callable, model, transform, operator,
  runtime, or geometry-specific algorithm.

The capability catalog exposes `robin` as declarable but non-executable.  No
qualified application is attached to it.

## Fail-closed execution boundary

`BoundaryKind.ROBIN` is not added to the existing FFT/DCT/DST map.  Current
public simulation lowering detects either homogeneous or nonhomogeneous Robin
faces and raises the structured rejection
`unsupported_robin_boundary` with canonical component/face context before
transform or runtime construction.

The older `ProblemSpec` spectral assembler also raises an explicit error that
Robin requires a qualified bounded-axis lowering.  It does not leak a mapping
`KeyError` and does not reinterpret Robin as DCT, DST, Dirichlet, or Neumann.

## Public contracts added

- `BoundaryKind.ROBIN`
- `StaticRobinCoefficients`
- `StaticRobinBC`
- `RobinFaceLaw`
- `StaticRobinBoundaryPolicy`
- `robin()`
- `LoweringRejectionCode.UNSUPPORTED_ROBIN_BOUNDARY`
- a non-executable `robin` capability-catalog entry

## Verification and scope

Tests cover finite coefficient validation, the degenerate-law rejection,
canonical metadata and SHA-256, homogeneous/nonhomogeneous identity,
deterministic face ordering, raw scaling identity, a generic evolved field
named `concentration`, independent face laws, periodic topology, exact
coverage, role rejection, dependency boundaries, public exports, discovery,
and structured pre-transform rejection.  Existing Dirichlet, Neumann,
Periodic, Plane, Channel, lifting, and public-catalog controls remain green.

No H100 work is needed for this tensor-free declaration slice.  There are no
new executable combinations, no numerical Robin residual claim, no Q finite
anchoring, no nonhomogeneous-Neumann feature, no default change, and no
benchmark claim.

P8.5.1 is complete.  P8.5.2 method-ADR and CPU-operator planning is eligible
for the next explicit request.  P8.5.2 implementation, P8.5 runtime work,
H100 submission, P8.6, Phase 9, compiled-runtime promotion, and production
default changes remain unauthorized.
