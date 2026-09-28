# Phase 8 final closure

Status: `PASS_PHASE8_CAPABILITY_EXPANSION_AND_INTEGRATED_H100_CLOSURE`.

Phase 8 is complete. It expanded the Phase 7 public architecture from two
qualified model--geometry combinations to four, added a stable public
capability catalog, introduced generic static Dirichlet lifting with strong
Plane Q-anchoring policies, and completed an internal finite-Q Robin
relaxation pilot. The final local package audit and the integrated
installed-wheel H100 matrix both passed.

## Qualified public capability matrix

The public catalog now contains two models, three geometries, eight boundary
policies, four executable model--geometry combinations, and six qualified
runtime paths. The executable applications are:

1. complete-stress Beris--Edwards on a Plane slab through
   `legacy_production` or `compiled_v2`;
2. legacy active-force active nematics in a rectangular Channel through
   `legacy_channel` or `compiled_channel_v2`;
3. complete-stress Beris--Edwards in a periodic box through
   `periodic_spectral`;
4. complete-stress Beris--Edwards in a rectangular Channel through
   `channel_complete_stress`.

Model, geometry, and boundary declarations remain orthogonal. Compilation
resolves a registered application before allocation, no registry lookup is
performed in the timestep, runtime fallback remains forbidden, and unsupported
combinations fail closed.

## Boundary capability at closure

The Plane complete-stress application supports static prescribed Q data and
the strong homeotropic and strong planar convenience policies through a
restart-qualified lifting route. This is a Plane Q capability, not a claim of
arbitrary prescribed fields on every geometry.

The finite-Q Robin relaxation work passed its internal numerical, restart,
conditioning, package, and H100 gates. It remains an internal partial pilot.
It is not connected to the public `Simulation` compiler, the complete
Beris--Edwards timestep, the Stokes solve, or a production runtime selector.
Nonhomogeneous Neumann lifting, dynamic/trainable boundary data, and general
public Robin execution remain future work.

## Integrated closure evidence

P8.6.1 verified the source and isolated installed-wheel catalogs, the complete
eight-case rejection matrix, and unchanged defaults. P8.6.2 then executed all
eight required H100 cases from a non-editable installed wheel. All cases were
finite, used the requested runtime without fallback, passed restart gates, and
passed the required negative identity and payload checks. Its archive contains
1204 verified checksum entries.

The Plane and Channel defaults remain `legacy_production` and
`legacy_channel`; compiled runtimes remain opt-in. No paper-benchmark,
steady-state, performance-promotion, or optimal-control result is claimed by
this architectural closure.

## Authorization

Phase 9 planning is now eligible, but Phase 9 implementation is not authorized
by this record. The finalization changes no runtime, timestep, numerical
operator, compiler registration, public API, checkpoint schema, output schema,
or production default.

The local finalization passed 10 targeted tests and the complete suite of 2427
tests plus 8 subtests. The archive source list includes the complete Phase 8
chain, but the existing verbatim PDF was deliberately not regenerated.

The machine-readable authority is
[phase_8_final_closure.json](phase_8_final_closure.json).
