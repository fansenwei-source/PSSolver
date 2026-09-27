# Phase 8 P8.5.3: generic Plane Robin scalar runtime state

Status: `PASS_P8_5_3_GENERIC_PLANE_ROBIN_SCALAR_RUNTIME_STATE`.

Baseline: `6b6995a`, the P8.5.2 recorded closure. Implementation commit:
`9d337feb17f9ad931fa756c34cbbf2a2455477c5`.

P8.5.3 connects the coefficient-specific P8.5.2 Robin operator to a
field-neutral Plane lowering plan, construction-owned operator cache,
runtime state, physical observation, and in-memory checkpoint semantics.
It deliberately remains an internal bounded-axis pilot: it is not a complete
three-dimensional timestep and is not reachable through the public
`Simulation` compiler, public runner, or a production runtime selector.

## Generic lowering contract

`pssolver.configuration.robin.lower_plane_robin_scalar_pilot` accepts one
registered evolved scalar field from an immutable `SimulationSpec`. The
qualified geometry is a cell-centered Plane slab with periodic axes 0 and 1
and bounded wall-normal axis 2. Both periodic axes must carry periodic laws;
both wall faces must carry static physical Robin laws.

The lowering is model-neutral. It imports no active-nematic module and knows
nothing about Q, directors, scalar order, molecular fields, or surface
energies. It records the source simulation identity, boundary-assignment
identity, field/component identity, geometry, domain, declared Fourier axes,
and the P8.5.2 bounded-axis plan. The periodic axes remain declared but are
not transformed by this pilot.

The lowering entry point intentionally remains an internal module API. It is
not re-exported from the frozen `pssolver.configuration` surface and is not
exported from top-level `pssolver`. Existing production lowering therefore
continues to reject unsupported Robin combinations fail-closed.

## Runtime state and observation

`PlaneRobinScalarRuntime` binds the operator once at construction and stores
the evolved scalar as a homogeneous remainder plus its synchronized bounded
modal representation. Physical observation is reconstructed on demand by
adding the static affine lift. Replacement of a physical observation updates
the remainder and bounded modes atomically and preserves the representation
ledger contract.

The runtime exposes bounded-axis Helmholtz apply and solve operations over
all independent x/y columns. It does not implement periodic FFTs, a model
right-hand side, a time integrator, a Stokes solve, or a complete timestep.
Consequently it makes no claim that a new executable model/geometry
combination exists.

## Cache and identity

`PlaneRobinOperatorCache` is construction-owned and keyed by the exact
lowering-plan identity, bounded-axis plan identity, geometry, full domain
shape and lengths, dtype, and device. Reusing the same plan returns the same
materialized operator. Changing Robin coefficients changes the plan and
cache identity. No cache lookup, root solve, basis inversion, or
factorization is placed in a timestep loop.

The qualified slice is CPU float64 only. The dense P8.5.2 operator remains a
reference backend; P8.5.3 adds no GPU kernel and requires no H100 test.

## Checkpoint semantics

The in-memory checkpoint records:

- lowering-plan and source simulation/boundary identities;
- raw lower and upper Robin coefficient identities;
- bounded-axis plan, materialized operator, and cache-key identities;
- geometry, field/component, dtype/device, and outward-normal convention;
- homogeneous remainder and bounded modal payload hashes;
- integrator progress and representation-ledger metadata.

Restore validates every identity, tensor shape/dtype/device, payload hash,
and physical/modal synchronization before mutating the target. Exact restore
recovers state and progress. Identity tampering and tensor-payload tampering
are rejected without target mutation. File-backed workflow/checkpoint
packaging remains outside this slice.

## Architecture boundary

The only new reviewed dependency edge is
`pssolver.configuration.robin -> pssolver.planning.robin`. The frozen v0.1
configuration exports and top-level public API remain unchanged. Generic
lowering/planning/runtime/operator modules contain no active-nematic imports.

P8.5.3 does not implement:

- finite-Q anchoring or a Landau-de Gennes surface energy;
- a Q-specific boundary specialization;
- periodic-axis execution or a complete PDE timestep;
- public `compile_simulation()` or `run_simulation()` support for Robin;
- a production runtime/default change;
- file-backed restart workflows;
- spatially or temporally varying Robin coefficients;
- prescribed nonzero Neumann flux, Channel, or curved boundaries;
- GPU execution, H100 qualification, or a paper benchmark.

P8.5.4 may now add the finite-Q anchoring specialization on top of this
generic scalar infrastructure. That work is eligible but is not implemented
or authorized by this record.

## Verification

Focused regression and P8.5.3 tests passed. The full local suite passed with
`2358 passed, 8 subtests passed`, with zero failures, skips, xfails, or
deselections.

The architecture archive source list includes these two P8.5.3 records. The
user-owned untracked PDF was not regenerated or changed.
