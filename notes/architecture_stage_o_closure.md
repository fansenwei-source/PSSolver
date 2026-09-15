# Architecture Stage O closure: retain production and the separated oracle

## Final decision

Stage O is closed without entering O.5 and without creating O.4.4. The Plane
`legacy_production` runtime remains the production default and rollback
reference. The `separated_canary` remains an opt-in numerical oracle,
architecture laboratory, and diagnostic comparison target; it is not a
production replacement. Channel, the generic solver, equations, boundary
conditions, numerical defaults, and benchmark `develop` remain unchanged.

This outcome separates two conclusions that must not be conflated. The
separated equations, timestep ordering, observation contract, and same-backend
restart are scientifically qualified: O.4 found exact initial-Q identity,
roundoff-scale six- and one-hundred-step differences, and byte-exact restart.
Production migration is nevertheless rejected because the benchmark-scale
runtime and memory evidence never qualified the canary as a replacement.

## Evidence chain

- O.4 (`12551bea`) was `B_neutral`. Its maximum numerical error was
  `3.169995072241099e-15`, but the frozen transform-count identity gate and
  R128 allocated-memory gate failed.
- O.4.1 (`6defce87`) corrected those contracts prospectively. At R320 the
  canary/legacy timestep ratio was `1.429558205567479`; peak allocated and
  reserved ratios were `1.589530529807474` and `1.328679331654841`.
  The excess was stable resident state, not a per-step leak.
- O.4.2.7 (`7e491231`) closed the observational storage-accounting audit.
  Inventory traversal was passive, allocator blocks reconciled, and no
  on-demand physical materialization was caused by observation.
- O.4.3 (`19ae28e3`) rejected generation-wide packed republication. It was
  bitwise exact but 3.016% slower at R320, with allocated and reserved memory
  increases of 26.30% and 28.15%.
- O.4.3.1 (`85e6a9a0`) retained conservative natural storage views as safe
  experimental machinery. It was bitwise exact and memory neutral, but its
  R320 ratio of `1.000066046754` was performance neutral.
- O.4.3.2 (`b668859f`) defined the producer-owned packing contract without
  claiming a performance result.
- O.4.3.3 (`f6ad4dc9`) applied producer-owned packing to molecular-field and
  stress outputs. It remained scientifically qualified and improved R320 by
  about 1.72%, but its ratio of `0.982766967768` missed the prospectively
  frozen `0.98` gate and its reserved memory increased.
- O.4.3.4 (`502c2b25`) attributed every remaining projected batch. The
  largest source, `algebraic.nematic_stress.dependencies`, used about
  `1.997209883%` of timestep time, below the frozen 3% screening signal.
  No source remained unattributed and no tensor reference was retained.

The exact report SHA-256 values and bounded scalar facts are recorded by
`pssolver.experimental.stage_o_closure`. The read-only CLI
`scripts_plane/close_plane_stage_o.py` emits that machine-readable record and
constructs no solver.

## Retention and rollback strategy

Keep the separated architecture because it is the clean responsibility model
and a qualified numerical oracle. Keep its neutral and rejected candidates as
immutable experimental evidence, not as runtime defaults. Do not delete the
historical copy/cat fallback or the legacy production adapter. Do not merge
producer-owned boundary packing into production on the strength of a 1.72%
gain that missed its frozen gate.

The projected-materialization optimization line is now closed. Refactoring Q
RHS, gradient, or dependency layouts would distribute risk across several
producers while the largest individually measured opportunity is below the
predeclared screening threshold. Reopening this line requires new evidence,
not a relaxed interpretation of O.4.3.3 or O.4.3.4.

## Stage P boundary

The next authorized action is a local Stage P diagnostic design. Stage P must
explain the remaining production/canary gap at the operator and kernel level,
with attention to transform scheduling, kernel-launch structure, compiled
graph boundaries, and repeated spectral operations. It begins as planning and
instrumentation only. It does not authorize H100 execution, an optimization,
a default change, another materialization layout, or a benchmark run.

Only a later, separately reviewed Stage P plan may define a new candidate and
prospective qualification gates.
