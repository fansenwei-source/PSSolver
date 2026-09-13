# Architecture Stage A: additive problem contracts

## Status and scope

Stage A starts from the qualified performance baseline
`afbc95e723c5ae9bb44079ae8979b51b883cef81`.  It adds immutable descriptions
of physical problems without connecting them to the production runtime.

The following existing modules remain untouched:

- `pssolver/solver.py`;
- `pssolver/Field.py`;
- `pssolver/PDEmodel.py`;
- `pssolver/integrator.py`;
- `pssolver/transforms.py`;
- Plane and Channel drivers;
- active-nematic execution kernels.

Consequently this stage cannot change equations, transforms, field ordering,
floating-point operation order, metadata, snapshots, or performance.

## Contract boundary

The new `pssolver.core` package contains descriptions rather than execution
objects:

```text
DomainSpec + GeometrySpec + ModelProtocol + NumericsConfig
                         |
                         v
                    ProblemSpec
                         |
                  future assembler
                         |
                         v
                  frozen SpectralPlan
```

The conceptual data flow above is not a Python import chain.  Physical models,
geometries, and numerical policies remain independent inputs.  A future
assembler will consume them and build a plan, preventing models from importing
Plane or Channel and preventing geometries from importing active-nematic
physics.

## Contracts added

### Physical boundaries

`BoundaryCondition` represents physical semantics.  Stage A supports only the
three homogeneous conditions already implemented by the current solver:

- `PeriodicBC`;
- `HomogeneousDirichletBC`;
- `HomogeneousNeumannBC`.

`BoundarySet` stores one condition per coordinate axis.  Conversion to legacy
string labels is deliberately outside the core contracts and is supplied by
the Stage B compatibility adapter.  No contract claims that a physical
boundary condition is intrinsically an FFT, DCT, or DST.

### Domain and geometry

`DomainSpec` owns tensor-product shape, physical lengths, axis names, and grid
placement.  `GeometrySpec` owns only axis topology: periodic or bounded.
Concrete `PeriodicBox`, `PlaneSlab`, and `RectangularChannel` constructors are
deferred to Stage B.

Geometry does not choose free-slip, no-slip, anchoring, or surface energy.
Those are physical boundary or closure choices that a future assembler will
combine with geometry.

### Fields

`FieldSpec` separates logical fields from stored scalar components.  Each
`FieldComponentSpec` has its own `BoundarySet`, allowing tangential and normal
velocity components, or tensor components with different parity, to compile
to different transform spaces.

Field roles are explicit:

- `EVOLVED`: advanced by a time integrator;
- `ALGEBRAIC`: recomputed from a constraint or elliptic solve, such as velocity
  and pressure in quasistatic Stokes flow;
- `DIAGNOSTIC`: derived for observation and output without feeding the model.

This replaces the misleading long-term use of `static` for fields that change
at every timestep but are not time-integrated.

### Numerical policy

`NumericsConfig` explicitly records precision, dealiasing, transform order,
projected-transform execution, spectral storage, and the Hermitian axis.  It
has no implicit Plane production preset.  Future geometry policies must create
qualified Plane, Channel, or reference configurations deliberately.

The contract rejects incompatible combinations before runtime, including:

- truncated projected transforms with no dealiasing;
- Hermitian storage without real-first execution;
- Hermitian storage without a non-negative packed axis;
- a Hermitian axis on a bounded geometry direction;
- a Hermitian axis supplied for full-complex storage.

### Models and problems

`ModelProtocol` currently freezes only stable declarative responsibilities:
model identity, field declarations, and parameter metadata.  The execution
protocol is deliberately deferred until the SpectralPlan facade exists, rather
than committing early to an operator API that would later need replacement.

`ProblemSpec` combines independent model, geometry, and numerical
specifications.  It validates dimensions, unique field/component names,
boundary/topology consistency, and Hermitian-axis geometry.  Its metadata is
JSON-compatible and contains no runtime tensors.

## Dependency rules

Future modules must obey these rules:

1. models may import core contracts, but not concrete geometries or transform
   implementations;
2. geometries may import domain and boundary contracts, but not physical
   models;
3. spectral planning may consume model, geometry, boundary, and numerical
   specifications;
4. transform backends execute a frozen plan and do not interpret active-matter
   terminology;
5. geometry-specific Stokes solvers consume a resolved spectral plan rather
   than parsing field boundary strings;
6. workflows are composition roots and may import models, geometries,
   integrators, runtime, and output modules;
7. no high-level specification parsing occurs inside the timestep hot path.

## Deferred decisions

Stage A intentionally does not add:

- nonzero Dirichlet or Neumann data;
- lower/upper wall values that differ;
- Robin, tau, lifting, or surface-energy plans;
- a symbolic/operator DSL;
- a stochastic API;
- a second time integrator;
- backend registration or non-PyTorch execution;
- concrete geometry presets;
- runtime field storage or model execution protocols;
- compatibility wiring into the existing solver.

These features should be introduced only after their preceding contracts have
been exercised by at least two real models or geometries.

## Stage B entry condition

Stage B may begin after the new contracts pass focused and complete CPU tests
and the branch is reviewed as additive-only.  Stage B should add concrete
`PeriodicBox`, `PlaneSlab`, and `RectangularChannel` specifications plus a
one-way adapter from physical boundary contracts to the current legacy labels.
It still should not replace the production transform or timestep path.
