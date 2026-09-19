# ADR 0008: separate tensor-free system declarations from execution

Status: accepted for Phase 2.

## Context

The Phase 2 component graph needs a geometry-neutral declaration of the
incompressible Stokes request.  The existing
`IncompressibleStokesSystemSpec` has the right scientific meaning and value
semantics, but it lives in `pssolver.execution.stokes` and imports
`AlgebraicSystemSpec` from `pssolver.execution.algebraic`.  That module also
contains Torch-dependent execution contracts.

Allowing `configuration -> execution` would make configuration parsing,
dry-run validation, documentation, and identity construction depend on an
execution layer.  It would also reverse the accepted declaration-to-execution
direction.  Copying the Stokes dataclass or hiding the dependency behind a
dynamic import would create two authorities for the same physical request.

The same placement problem exists one level below Stokes.
`AlgebraicSystemSpec` and `AlgebraicUpdatePhase` are tensor-free declarations,
but their current module mixes them with device/runtime contracts.  The
architectural boundary should reflect their meaning rather than the contents
of one historical source file.

## Decision

Introduce a narrow provisional `pssolver.systems` layer for geometry-neutral,
tensor-free PDE subsystem requests and generic algebraic-system declarations.
It is not a symbolic equation language, plugin framework, solver registry, or
miscellaneous declarations package.

The intended modules are:

```text
pssolver.systems.algebraic
    AlgebraicUpdatePhase
    AlgebraicSystemSpec

pssolver.systems.stokes
    INCOMPRESSIBLE_STOKES_CAPABILITY
    PressureGauge
    TangentialZeroModePolicy
    IncompressibleStokesSystemSpec
```

`systems` may depend only on the standard library, `pssolver.core`, and other
`pssolver.systems` modules.  It is included in the executable tensor-free
layer gate.  It contains no Torch/NumPy tensors, transform implementations,
geometry-specific solver, backend, execution state, workspace, registry,
runtime, workflow, or application code.

Typed-to-generic conversion from `IncompressibleStokesSystemSpec` to
`AlgebraicSystemSpec` remains a declaration normalization step within
`systems`.  Geometry capability selection, discretization lowering,
device binding, and solving remain in planning and execution layers.

The dependency direction becomes:

```text
core
  -> geometries / model declarations / system declarations
  -> planning
  -> backends / operators / linear solvers
  -> execution / integrators / runtime
  -> workflows / I/O
  -> applications
```

Configuration, models, planning, and execution may consume system
declarations where they have a real use.  `systems` never imports those
higher layers.  In particular, this decision does not authorize
`configuration -> execution` or `execution -> models`.
Each `-> systems` permission is added to the executable ratchet only in the
same commit as its first real import; this ADR does not pre-authorize unused
layer-wide edges.

## Compatibility migration

The move is performed mechanically and in this order:

1. Characterize the current algebraic and Stokes declaration contracts in
   micro-phase P2.1S:
   exports, object identity, dataclass flags and fields, signatures, defaults,
   enum values, validation, metadata, algebraic round-trip, pickle global
   paths, and both successful and rejected instance-pickle behavior.
2. Move `AlgebraicUpdatePhase` and `AlgebraicSystemSpec` to
   `pssolver.systems.algebraic`.
3. Move the four Stokes declarations to `pssolver.systems.stokes` without
   changing fields, defaults, validation, metadata, or round-trip behavior.
4. Keep `pssolver.execution.algebraic`, `pssolver.execution.stokes`, and
   `pssolver.execution` as compatibility import paths for the moved names.
5. Require every old and canonical import path to resolve to the same class or
   enum object.  A wrapper, subclass, duplicate dataclass, protocol-only
   substitute, or dynamic-import bridge is not acceptable.
6. Add `systems` to the dependency and tensor-free ratchets before a
   configuration component imports it.  The first `systems` implementation
   and its ratchet rules land together.
7. Only then add `PlaneBerisEdwardsPhysicsSpec` and continue the pure facade
   decomposition.

Moving the canonical definition changes `__module__` for newly constructed
objects.  That is an implementation-identity change within a provisional
architecture API, not a scientific or discretization change.  The old module
paths remain importable and old pickles that reference those globals must
load in the new version.  Forward loading a new-version pickle in v0.1.2 is
not promised.  Production checkpoints do not pickle these concrete classes.

No production runtime, dispatch policy, solver implementation, numerical
default, metadata schema, checkpoint format, output, or hot-path operation is
changed by this declaration extraction.

## Rejected alternatives

### Put Stokes declarations in `pssolver.core`

This has a small mechanical diff, but Stokes is one physical subsystem rather
than vocabulary required by every solver.  Repeating the pattern for Poisson,
Navier--Stokes, elasticity, phase field, and electromagnetism would turn core
into a model catalogue.

### Put shared Stokes declarations in `pssolver.models`

The physical ownership is plausible, but execution currently cannot depend on
models, while legacy model declarations already depend on execution.  A
compatibility re-export would therefore create or encourage a bidirectional
models/execution dependency.  Stokes is also shared by multiple coupled
models rather than owned by active nematics.

### Allow `configuration -> execution`

This preserves the current file location at the cost of making the temporary
placement permanent.  It couples tensor-free identity construction to runtime
dependencies and creates a new exact architecture debt, so it is rejected.

### Duplicate, subclass, or dynamically import the Stokes type

Two nominally similar classes fail strict `isinstance` checks, dataclass
equality, exact type dispatch, and object-identity assertions.  Indirection
would hide rather than remove the dependency.  Compatibility is therefore an
alias to one canonical object, not a second definition.

## Qualification gates

The extraction must pass all of the following before the Phase 2 physics
aggregate is added:

- exact old/canonical import-object identity;
- unchanged dataclass fields, signatures, defaults, validation, equality,
  hashing, representation, metadata, and algebraic round-trip;
- unchanged strict `isinstance` behavior and enum-member identity at existing
  model and solver-factory boundaries;
- unchanged generic-system metadata and restart-provenance hashes;
- old-global pickle load, canonical round-trip for every currently pickleable
  instance, and unchanged rejection for currently unpickleable instances;
- explicit characterization of the existing `AlgebraicSystemSpec` instance
  pickle failure caused by its normalized `MappingProxyType` parameters;
- no new stable package-root export;
- tensor-free and import-boundary tests with no new legacy exception;
- the complete CPU suite, wheel import smoke, and console-entry smoke;
- `git diff --check` and a clean worktree.

Because the extraction changes no runtime consumer or numerical path, it does
not require an H100 qualification by itself.  The existing Phase 2 final H100
gate remains required after consumer migration.

## Consequences

Phase 2 gains one explicit declaration layer, but avoids contaminating core or
coupling configuration to execution.  Plane and future Channel/periodic
compositions can share one Stokes request while selecting different
geometry-specific solvers later.  The same layer can host a small number of
real cross-model system requests, but a second genuine consumer is still
required before any such interface becomes stable public API.
