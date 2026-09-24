# Phase 7 P7.7.1: geometry-neutral equation-system declarations

Status: `P7_7_1_COMPLETE_P7_7_2_NOT_AUTHORIZED`.

Parent baseline: `e7f1685d2493ed965f6b6652513a286a28b4fbda` on
`next/pssolver-v0.2.0-architecture`.

Classification:
`PASS_P7_7_1_GEOMETRY_NEUTRAL_EQUATION_DECLARATIONS`.

P7.7.1 introduces a tensor-free scientific declaration seam. It changes no
equation, boundary condition, geometry, transform, timestep, runtime,
workflow, checkpoint, output, or production default. Existing Plane and
Channel applications do not consume the new declarations yet.

The machine-readable authority is
[phase_7_p771_equation_system_declarations.json](phase_7_p771_equation_system_declarations.json).

## Why a new field declaration was necessary

The existing `pssolver.core.FieldSpec` correctly represents a field after its
physical boundary conditions have been assigned. Each scalar component owns a
`BoundarySet`. That makes it suitable for a validated `ProblemSpec`, but not
for a geometry-neutral model declaration.

P7.7.1 therefore introduces `EquationFieldSpec`. It owns only:

- the logical field name;
- the field role (`evolved`, `transient`, `algebraic`, or `diagnostic`);
- ordered scalar component names.

It contains no geometry, face, boundary, basis, transform, storage, tensor,
backend, or device. Later composition attaches physical boundary laws to
these component names and geometric faces. Lowering then selects a compatible
FFT/DCT/DST, lifting, tau, derivative, and geometry-specific solver path.

This prevents an important category error: a model declaration must not say
that Q is “DCT” or “Neumann in z” before a geometry and wall assignment exist.

## Generic declaration layer

The new canonical module is `pssolver.systems.equations`.

### `EquationFieldSpec`

Declares a boundary-free logical field and its scalar components.

### `EquationTermSpec`

Declares one evolution or constitutive term:

```text
name
capability
ordered output components
ordered dependencies
finite JSON-compatible physical parameters
```

A capability is a scientific/numerical requirement, not a concrete
implementation. For example, `beris_edwards_molecular_field` does not select
a transform or GPU kernel.

### `EquationSystemSpec`

Aggregates:

- physical model name and variant;
- boundary-free fields;
- evolution laws;
- constitutive laws;
- tensor-free algebraic subsystem requests;
- physical parameter metadata;
- diagnostic declarations;
- model-owned initial-condition families;
- deterministic canonical SHA-256 identity.

The composition fails closed unless:

1. logical field and component names are unique;
2. every dependency names a declared component;
3. evolution terms produce evolved fields;
4. constitutive terms produce transient fields;
5. algebraic systems produce algebraic fields;
6. every non-diagnostic component has exactly one producer;
7. evolution, constitutive, and algebraic producer names are unique.

`EquationSystemSpec` deliberately does not implement the executable
`ModelProtocol`. It cannot execute without boundary assignment,
geometry-aware planning, and device binding.

## Two distinct physical declarations

The canonical active-nematic declaration module is
`pssolver.models.active_nematics.equation_systems`. It imports only core field
roles and tensor-free system declarations. It does not import Torch,
geometries, transforms, solvers, runtime, workflows, or applications.

### Complete-stress Beris--Edwards

`CompleteStressBerisEdwardsEquationRequest` declares the physical path used by
the qualified Plane application:

```text
Q
  -> Q gradient
  -> molecular field
  -> algebraic reactive + active stress
  -> distortion stress
  -> complete nematic force
  -> incompressible Stokes -> velocity, pressure

velocity -> velocity gradient
Q + velocity + both gradients -> complete Q evolution
```

The declaration contains 59 scalar components across nine logical fields. It
records the one-constant `L1`, rotational viscosity, Landau--de Gennes
coefficients, flow alignment, active amplitude, beta, complete active
prefactor, IMEX split, and the tensor-free Stokes request.

Its force law is explicitly:

```text
complete_one_constant_nematic_stress
```

Its initial-condition family is `extruded_defect_gas`. This is a model-owned
family name only; no Plane grid or boundary is attached.

### Legacy active-force active nematics

`LegacyActiveForceEquationRequest` declares the physical path used by the
qualified Channel application:

```text
Q -> active force divergence
  -> incompressible Stokes -> velocity, pressure

Q -> Q gradient
velocity -> velocity gradient
Q + velocity + both gradients -> legacy Q evolution
```

The declaration contains 36 scalar components across six logical fields. It
records the rho-derived Landau--de Gennes coefficients, elastic constant,
activity, beta, flow alignment, IMEX split, and tensor-free Stokes request.

It intentionally does not declare molecular-field, reactive-stress, or
distortion-stress fields. Its force law is explicitly:

```text
active_force_divergence_only
```

Its initial-condition family is `aligned_x_smooth_noise`.

These are two variants of the active-nematic family, not one model with two
geometry labels. Their canonical identities are necessarily different.

## Compatibility adapters

`pssolver.configuration.active_nematics_equation_adapters` contains two pure
adapters:

```text
PlaneBerisEdwardsRunComponents
  -> declare_plane_complete_stress_equation_system
  -> EquationSystemSpec(complete_stress_beris_edwards)

ChannelRunComponents
  -> declare_channel_active_force_equation_system
  -> EquationSystemSpec(legacy_active_force_active_nematics)
```

The Plane adapter preserves the existing material request, resolved `L1`,
resolved activity amplitude `zeta`, and Stokes request. The Channel adapter
preserves rho and its derived coefficients, elastic constant, activity, beta,
flow alignment, and Stokes request.

Neither adapter reads geometry or boundaries. Neither constructs a runtime.
Neither existing application or runtime imports the adapter. This makes the
new declarations auditable without changing the qualified numerical path.

## Dependency rule

P7.7.1 authorizes exactly two model-to-system import edges:

```text
pssolver.models.active_nematics.equation_systems
  -> pssolver.systems.equations

pssolver.models.active_nematics.equation_systems
  -> pssolver.systems.stokes
```

The architecture test records these exact edges rather than broadly allowing
every model module to import every system module. The configuration adapter
has one separately recorded edge to `pssolver.systems.equations`.

## Public and runtime surface

P7.7.1 adds no package-root export. Direct leaf-module imports are provisional
architecture surfaces. It changes no signature in:

- `PlaneBerisEdwardsRunSpec`;
- `ChannelActiveNematicRunSpec`;
- either application entry point;
- either runtime facade;
- either workflow or checkpoint format.

Because no runtime imports these declarations, this slice adds no per-step
dispatch, allocation, tensor copy, GPU memory, transform call, or performance
cost.

## Next slice: P7.7.2

P7.7.2 is **not authorized by this result**. When separately authorized, it
should introduce physical `BoundaryAssignment` declarations and a tensor-free
`SimulationSpec` that composes:

```text
EquationSystemSpec
+ GeometrySpec
+ BoundaryAssignment
+ NumericsSpec
+ TimeIntegrationSpec
+ ExecutionSpec
+ WorkflowSpec
+ invocation/provenance identity
```

P7.7.2 must remain disconnected from runtime construction. Basis and solver
selection belong to the following capability-lowering slice. New boundary
physics, production default changes, and Phase 8 remain out of scope.
