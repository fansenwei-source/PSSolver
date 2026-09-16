# Architecture Stage Q.6.2: native-segment shadow candidate

Stage Q.6.2 implements the candidate authorized by the frozen Q.6.1 design.
It is limited to the experimental Plane Beris--Edwards shadow runtime and does
not alter production defaults, the generic solver, Channel, equations,
boundary signatures, or the integrator.

The scheduler receives an exact allow-list of three semantic sources:

- `algebraic.nematic_stress.dependencies`
- `explicit_rhs.dependencies`
- `explicit_rhs.outputs`

For those sources, each boundary-compatible batch is partitioned into maximal
producer-native contiguous-storage segments.  Each segment is transformed
independently; singleton tensors are valid one-component segments.  The
scheduler never concatenates segments, copies them into a workspace, retains
their tensor references, or silently falls back to `copy_cat`.  Result pieces
are restored to the existing consumer component order by index metadata.

The candidate deliberately changes transform partitioning and can increase
kernel launches.  Qualification therefore measures three balanced R320 H100
trials per role, a six-step Q/u/p trajectory, per-source copy and segment
counters, peak allocated and reserved memory, and the observed timing cost.
Numerical equivalence is gated at relative L2 <= 1e-12.  The experiment cannot
promote a production default; it can only authorize a later architecture
decision when the copy elimination is both structurally real and measurably
beneficial.
