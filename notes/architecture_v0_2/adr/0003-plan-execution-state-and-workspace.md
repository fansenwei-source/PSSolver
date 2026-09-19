# ADR 0003: plan, execution, state, and workspace ownership

Status: accepted.

## Context

The v0.1.2 `Fields` object combines logical layout, storage, physical/spectral
representations, transforms, differential operations, and synchronization.
The separated runtime made ownership more explicit but introduced persistent
state and dynamic scheduling overhead.  Control code currently obtains a
functional step by replacing internal buffers.

## Decision

Four responsibilities remain distinct:

- `DiscretizationPlan` is immutable and tensor-free.
- `ExecutionPlan` is immutable after binding and owns device resources,
  selected implementations, direct callables, and a workspace plan.
- `FieldLayout` is immutable and maps semantic components to packed storage.
- `RuntimeState` is the sole owner of mutable evolved values, clock,
  representation validity, and declared persistent algebraic state.

Generation-local molecular fields, gradients, stresses, forces, and temporary
transform buffers belong to a bounded workspace.  Diagnostics hold no mutable
cross-generation tensor reference.

Physical and spectral representations have explicit current/stale generation
state.  The debug/reference path may check generation tokens; the optimized
path lowers validated transitions to a fixed program.

## Hot-loop rule

No string parsing, dependency discovery, registry lookup, JSON operation, or
metadata assembly occurs inside a timestep.  Logical mapping ergonomics are
resolved to integer handles, slices, grouped buffers, and direct callables
before execution.

## Functional execution

An optimized in-place executor and a future differentiable functional
executor may use different mutation policies.  They must share the same
scientific and discretization plan.  The production path is not required to
retain an autograd graph.

## Consequences

State and checkpoint contracts become inspectable without forcing a slower
per-component runtime.  Internal packing remains private and may evolve as
long as semantic state and qualified outputs are preserved.
