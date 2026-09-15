# Architecture Stage O.4.3.3: producer-packed H and stress

## Decision scope

Stage O.4.3.1 showed that consuming naturally adjacent tensor views is safe and
memory-neutral, but it removed only one of nine multi-component copy/cat
batches per steady timestep and produced no measurable R320 speedup.  Stage
O.4.3.2 therefore defined a producer-owned packed-output contract.  This stage
adapts exactly two Plane pointwise producers to that contract:

- the spectral bulk molecular-field producer;
- the complete algebraic plus distortion stress producer.

The Q explicit right-hand side, Q/velocity gradient producers, transform
ordering, field lifetimes, Stokes implementation, equations, and all production
defaults remain unchanged.

## Storage contract

The opt-in `boundary_packed` layout returns one contiguous tensor with shape
`(component, batch, *grid)`.  Packing is expressed inside the pointwise kernel,
so the compiled CUDA candidate can allocate the producer result in its final
component-major storage.  No Python-side post-kernel `stack` or `cat` is
permitted.

Molecular-field components use canonical H order because all five have the same
Plane boundary signature.  Stress components are stored in the exact order
compiled by the boundary-signature scheduler: all nine wall-even algebraic
components, the five wall-even distortion components, then the four wall-odd
distortion components.  Each scheduler group is therefore one contiguous view.

The packed tensor is generation-local.  It is neither checkpointed nor retained
across algebraic generations.  The existing `component_mapping` path remains
the default and fallback.

## Qualification

The H100 baseline is the qualified O.4.3.1 safe-view configuration:
`deferred_stack + contiguous_storage_view + component_mapping`.  The candidate
changes only the H/stress producer layout to `boundary_packed`.

The bounded gate requires:

- identical production Q0 identity and 100-step Q/u/p relative L2 no larger
  than `1e-10` (pressure compared after demeaning);
- producer provenance proving in-kernel producer ownership, no post-kernel
  stack/cat, and no cross-generation reuse;
- more zero-copy view batches and fewer copy/cat batches than the baseline;
- three balanced trials at R128 and R320;
- R320 mean timestep ratio at most `0.98`, worst paired ratio at most `1.02`,
  and at least two of three paired trials faster;
- peak allocated memory ratio at most `1.03` and safety limits of `1.03` for
  mean timestep and `1.10` for peak allocated memory.

An `A_recommended` result authorizes only the next architecture decision.  It
does not authorize a production default change.
