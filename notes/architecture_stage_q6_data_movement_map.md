# Architecture Stage Q.6: algebraic data-movement map

Stage Q.6 is a read-only diagnostic that implements the exact scope selected
by Stage Q.5. It joins the frozen O.4.3.4 measurements to current static
runtime contracts. It does not construct a solver, read trajectory arrays,
change a timestep, implement a candidate, or alter a production default.

The three measured boundaries are:

1. `algebraic.nematic_stress.dependencies`: missing spectra for physical
   stress dependencies are grouped by projected-boundary signature, copied
   into an inverse-transform batch, transformed, split into views, and exposed
   by component name to the stress solver.
2. `explicit_rhs.dependencies`: transient algebraic spectra needed by the
   physical explicit-RHS island follow the same inverse-transform batching
   mechanism and are cached by component name for the current generation.
3. `explicit_rhs.outputs`: independent physical RHS outputs are grouped and
   copied into forward-transform batches; the resulting spectra are restored
   to evolved-component order and stacked for the integrator.

## Main diagnosis

All three sources share a root mechanism: independently owned component
tensors cross a boundary at which the transform scheduler requires a packed,
boundary-signature-homogeneous leading dimension. `copy_cat` currently bridges
that layout discontinuity.

This is not evidence that projected transforms should be removed. The
transforms and their parity groups remain required numerical operations. It is
also not evidence that allocator reuse alone is useful: Stage Q.4 already
showed that a preallocated workspace retains the copy while increasing memory
and failing the performance gate.

The measured source times are nested subregions. Operator, copy-kernel, and
transform durations may overlap, so their fractions are not additive and are
not a predicted speedup.

## Bounded continuation

Stage Q.6 may authorize only a Q.6.1 design review for a
`boundary_signature_native_handoff_contract`. That design must prove ownership,
generation lifetime, boundary-signature separation, consumer access without
hidden repacking, preserved publication order, and bounded peak memory before
any implementation is permitted.

Q.6 does not authorize a candidate implementation, solver execution, equation
change, production promotion, or default change. The original R--W roadmap is
unchanged; any Q.6.x work remains a bounded diagnostic/design branch before
resuming Stage R.
