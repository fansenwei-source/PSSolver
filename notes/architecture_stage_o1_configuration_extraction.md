# Architecture Stage O.1: pure Plane configuration extraction

## Outcome

Stage O.1 extracts the current production Plane Beris--Edwards configuration
without changing its numerical runtime.  `Plane_beris_edwards_stokes.py`
continues to construct and execute only the qualified legacy production path.
There is no runtime selector, canary import, production-default change, Channel
migration, or generic-solver migration in this stage.

## Configuration authority

`PlaneBerisEdwardsRunSpec` is the one immutable authority produced by either
the production CLI parser or the programmatic factory.  It contains the
domain, workflow, initial condition, zero-mode choice, numerical choices,
output policy, resolved refresh policy, and physical boundary declaration.
The production script consumes the same attribute interface as before, so the
solver-construction and timestep code retain their established operation
order.

The metadata contains one additive `configuration` identity with:

- schema version;
- fully qualified authority name;
- fixed `legacy_production` runtime path;
- SHA-256 of the canonical resolved run specification.

It does not expose a candidate runtime choice.  The separate-canary selector
belongs to Stage O.2 and remains unauthorized here.

## Pure benchmark preset

The Shendruk-inspired mapping has moved out of module-global production code
into `pssolver.presets.shendruk`.  The pure resolver preserves:

```text
activity_ratio = (A / H)^2 = zeta / K
q_eq = 3 S_eq / 2
L1 = K / (2 q_eq^2)
```

Both `paper-window` and `fixed-k` parameterizations retain their previous
arithmetic and validation.  The resolver creates no tensor or solver.

## Physical boundary declaration

The free-Q/free-slip Plane boundary conditions are now expressed as immutable
core `BoundarySet` objects.  The legacy string tuples are produced only by the
one-way legacy-boundary adapter.  The resolved physical conditions remain:

- Q and tangential velocity: periodic, periodic, homogeneous Neumann;
- normal velocity: periodic, periodic, homogeneous Dirichlet;
- pressure modal multiplier: periodic, periodic, homogeneous Neumann;
- odd-z distortion terms: periodic, periodic, homogeneous Dirichlet.

This is a representation cleanup, not a new wall model.

## Provenance and validation tooling

The production implementation-file set is now owned once by the configuration
module and shared by the production driver, validation planner, and defect-core
analyzer.  New configuration, preset, geometry, physical-BC, and adapter source
files are included in implementation provenance.

Before and after the extraction, a local two-step float64/eager CPU trajectory
was run on an `8 x 8 x 6` Plane grid with spectral refresh disabled.  The
initial 2D arrays and every saved Q, velocity, and pressure array were
byte-for-byte identical.  Dry-run scientific metadata was canonically equal
after excluding the necessarily changed implementation provenance and the new
additive configuration identity.

The final local validation results were:

- 198 focused configuration/CLI/validation/analyzer tests passed;
- 218 architecture-chain and Stage-N qualification tests passed;
- 859 complete repository tests and 8 subtests passed;
- no test failure or skip was present in the local environment.

## Next boundary

The only next architectural action is Stage O.2: introduce a narrow Plane
runtime protocol and an explicit opt-in separated canary while retaining
`legacy_production` as the omitted-selector behavior.  O.1 does not authorize
that implementation or any production promotion by itself.
