# Phase 2 local qualification

Status: `PASS_PHASE2_LOCAL_H100_PENDING`

Phase 2 implementation candidate:
`4260480989169e817fba4af1949c15c44223b248`.

Release comparison baseline:
`v0.1.2` / `4fa614616e9d67c98be52c150eaf303a0c80b5c1`.

The machine-readable record is
[phase_2_local_qualification.json](phase_2_local_qualification.json).

## Provenance closure

The final audit found exactly two imported package facades that could select
production declarations but were not themselves hashed:

- `pssolver/core/__init__.py`;
- `pssolver/geometries/__init__.py`.

They are now part of the 54-file implementation source inventory.  The audit
test walks every inventoried Python source and rejects any imported PSSolver
package facade that is not itself inventoried.  This changes implementation
provenance only; schema v1, scientific configuration, runtime selection,
floating-point operations, and production defaults are unchanged.

## CPU and numerical gates

The focused provenance/configuration/serializer/validator gate passed 357
tests.  The complete CPU suite passed 1623 tests and 8 subtests.

Both `legacy_production` and `separated_canary` were run at `8 x 8 x 8` for
four CPU float64 eager steps from the release baseline and the Phase 2
candidate.  For each same-runtime pair, all 34 non-JSON trajectory,
initial-condition, diagnostics, checkpoint-array, and completion files were
byte-identical.  Dry-run metadata was canonically equal after removing the
prospectively allowed implementation-provenance field, and dry-run allocated
no output directory.

At step 2, release checkpoints were resumed by the candidate and candidate
checkpoints were resumed by the release.  Both directions passed for both
runtime paths; final step-4 `Q/u/p` was byte-identical to the corresponding
continuous trajectory.

## Package gate

Fresh wheel and sdist archives contain both closure facades.  A new isolated
environment imported the installed wheel rather than a source checkout,
preserved canonical policy object identity, completed a console dry-run
without output, and completed one CPU step through both runtime paths.

## Remaining Phase 2 gate

Local qualification is complete.  Phase 2 is not closed until one final,
single-job H100 task compares v0.1.2 with the implementation candidate:

- three balanced profile trials per role at R128 and R320;
- mean and median timestep ratios no greater than 1.02;
- peak allocated and reserved ratios no greater than 1.02;
- identical transform counts and zero fallback/graph-break/error counters;
- 100-step same-runtime production trajectories with byte-identical `Q/u/p`;
- same-runtime cross-version restart and output-schema checks.

Passing this gate closes Phase 2.  It does not change a production default,
promote `separated_canary`, merge into `develop`, or authorize Phase 3.
