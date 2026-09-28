# Phase 9 P9.1: periodic friction, z-invariant Q, and functional declarations

Status: `PASS_P9_1_FUNCTIONAL_PREREQUISITES`.

Baseline: `30c1d4c92045f0cb5db6ef58c3e0d49feb598feb` on
`next/pssolver-v0.2.0-architecture`.

P9.1 closes the prerequisites named by the P9.0 freeze.  It changes no
equation, timestep, production default, durable checkpoint schema, or stable
package-root API.  Its machine-readable authority is
[phase_9_p91_functional_prerequisites.json](phase_9_p91_functional_prerequisites.json).

## Public periodic friction branch

`CompleteStressBerisEdwards` now accepts an explicit friction coefficient and
an explicit tangential zero-mode policy.  Its defaults remain exactly
`friction=0` and `zero_mean`.  The public periodic compiler accepts exactly two
valid pairs:

- `zero_mean` with zero friction removes all uniform velocity and force modes;
- `friction` with positive friction retains the uniform velocity mode and
  resolves it through Brinkman drag.

Pressure remains zero mean in both cases.  Invalid pairs fail in the typed
Stokes declaration before allocation.  Compiler, application, and runtime
metadata now record the effective policy, friction, and uniform-mode action.
The Plane and Channel public compilers were not broadened by this slice.

This is a public connection to an already implemented and manufactured-tested
periodic Stokes branch; it does not introduce a new Stokes formula.

## z-invariant Q contract

The active-nematic model package now provides two pure helpers:

- `embed_z_invariant_planar_q` maps `(Qxx,Qxy,Qyy)` with layout
  `(3,batch,nx,ny)` to the canonical 3D compact layout
  `(Qxx,Qxy,Qxz,Qyy,Qyz)` with shape `(5,batch,nx,ny,nz)`;
- `project_z_invariant_planar_q` fail-closed checks exact z invariance and
  exactly zero `Qxz` and `Qyz`, then returns the planar components.

Neither helper converts dtype or device.  The Torch path remains
differentiable.  Tests establish exact embedding/projection round trips,
rejection of non-planar or z-dependent inputs, a complete public periodic
step with `Nz=1`, and preservation of the planar z-invariant subspace for a
multi-plane run.

This contract embeds the planar subset of the existing five-component 3D Q
convention.  It does not add a separate 2D Q theory or silently average a
non-invariant 3D field.

## Provisional functional declarations

The new `pssolver.functional` namespace defines declarations only:

- exact tensor, state, control-field, observation, capability, runtime
  identity, and construction-request records;
- an explicit `FunctionalRuntimeProtocol` and factory protocol;
- the frozen input-state observation timing;
- flat-tuple state with explicit batch axes and layout meanings;
- exact dtype/device validation with no implicit conversion;
- model-owned bounds, grid placement, broadcast rules, equation injection,
  and dealiasing identity for control fields;
- JSON scientific/discretization/execution/state-layout identity;
- fail-closed capability defaults: no pure-step, replay, differentiability,
  checkpoint bridge, JVP, or VJP claim until later qualification.

The protocol includes `step_index` in `step` and `step_and_observe`.  The first
construction request accepts batch size one only; larger batches remain a
future explicitly declared capability.  The namespace is intentionally not
re-exported from `pssolver` while provisional.

P9.1 does not implement a functional runtime, functional timestep,
observation computation, activity-control injection, gradient, adjoint,
optimizer, checkpoint conversion, or production/functional consistency
validator.  The frozen legacy `pssolver.control` oracle is unchanged, and no
new PSSolver dependency on an independent consumer exists.

## Verification and next slice

Focused CPU tests cover the new contracts plus the existing periodic
application, periodic Stokes manufactured tests, and typed public declarations.
The complete local suite and `git diff --check` are recorded in the machine
record.  No H100 job was needed or submitted because P9.1 changes only public
connection logic, pure tensor helpers, and declarations; P9.2 is the first
slice that will introduce a new executable functional path.

P9.1 is complete.  P9.2 is next: implement the batch-one periodic
complete-stress activity-control functional runtime against these declarations,
without changing production defaults or extending the legacy control oracle.
