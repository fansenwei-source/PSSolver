# Architecture Stage O.4.3: packed algebraic publication

## Decision

Stage O.4.2 closed the CUDA storage-accounting gate and identified algebraic
representations plus projected-transform batch assembly as the first bounded
optimization target. Stage O.4.3 therefore changes neither the equations nor
the spectral discretization. It tests one opt-in storage-layout candidate:

- publish stored algebraic spectra into one declaration-ordered packed tensor;
- publish transient spectra into a separate, generation-local packed tensor;
- let the transform scheduler order transient storage by its complete boundary
  compatibility partition;
- assemble a transform batch as a zero-copy view only when all selected tensors
  are consecutive contiguous views of exactly one storage;
- retain the historical `torch.cat` path as the mandatory fallback.

Stored and transient spectra deliberately use separate allocations. A static
stored-field view must not keep the much larger transient allocation alive
after the algebraic generation is invalidated.

## Safety invariants

The candidate preserves:

- algebraic system and component execution order;
- boundary signatures and transform order;
- physical and spectral shapes, dtypes, and devices;
- one-generation transient lifetime;
- no cross-generation tensor reuse;
- the existing `deferred_stack` and `copy_cat` behavior as defaults;
- the production Plane runtime default and all Channel/generic solver paths.

The zero-copy path requires matching shape, stride, dtype, device, strided
layout, common storage identity, and exact adjacent storage offsets. Autograd,
empty/scalar, conjugate/negative, non-contiguous, non-adjacent, and distinct
storage inputs all fall back to `torch.cat`.

## Qualification boundary

Stage O.4.3 is an H100 A/B qualification, not a production promotion. It binds
the frozen Stage O.4.2 accounting report, compares the historical canary
storage path against the packed/view candidate, and requires:

- a common immutable production Q0;
- a 100-step Q/u/p relative-L2 gate of `1e-10`;
- three balanced trials at R128 and R320;
- proof that zero-copy batches were exercised and reduced copy-cat batches;
- at least a 2% R320 mean timestep improvement;
- paired timing and peak-allocated-memory non-regression.

An `A_recommended` result permits only a later Stage O.4.4 architecture
decision. It does not change any production default.
