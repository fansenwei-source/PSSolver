# ADR 0002: problem declarations and lowering

Status: accepted.

## Context

The current Plane run specification combines scientific parameters, geometry,
boundary data, integration, execution policy, initialization, and output.
Model packages also contain some concrete transform and Plane solver
dependencies.  These choices make it difficult to reuse one model across
geometries without copying assembly code.

## Decision

The target declaration flow is:

```text
ModelSpec + GeometrySpec + BoundaryAssignment + NumericsSpec
    -> ProblemSpec
    -> tensor-free DiscretizationPlan
    -> device-bound ExecutionPlan
```

Models declare logical fields, parameters, equations, constitutive laws,
algebraic requirements, diagnostics, and model-owned initial conditions.
Geometries declare topology and metric information.  Boundary assignments are
independent physical inputs.  Planning binds these declarations and lowers
them to basis, parity, derivative, lifting, dealiasing, zero-mode, and solver
capability decisions.

The first implementation uses typed Python protocols and dataclasses.  It does
not introduce a symbolic equation language.

## Consequences

- One Beris--Edwards declaration can later be bound to Plane and Channel.
- DCT/DST remain numerical mechanisms rather than public physical BC names.
- Nonzero Dirichlet, Robin, and surface anchoring can be introduced through
  lowering without changing geometry identity.
- Construction performs more validation, but the timestep performs less
  dynamic interpretation.

## Compatibility

`PlaneBerisEdwardsRunSpec` remains the v0.1.2 compatibility facade while its
responsibilities are split internally in a later phase.  Phase 0 changes no
configuration schema or CLI behavior.

