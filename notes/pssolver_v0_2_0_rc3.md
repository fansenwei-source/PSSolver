# PSSolver 0.2.0rc3 release candidate

PSSolver 0.2.0rc3 is a correctness and independent-consumer compatibility
follow-up to `v0.2.0rc2`.  It does not expand the public capability matrix,
change functional protocol version 1.0, or promote a compiled runtime to a
production default.

## Provider correction

The Channel functional pressure transpose now uses the same unresolved
x-Nyquist subspace as the production pressure operator on even grids.
Fixed-cap pressure solves also reject operator breakdown and residual
amplification.  H100 Job 10857845 qualified this correction at
`165d3b6a4007f8ebd46f3455ceb2a4d57667653c`; forward Q, velocity, and pressure
outputs remained byte-identical to rc2 while the erroneous transpose mismatch
fell from approximately 0.407 to `1.01e-16`.

## Explicit declaration migration

The stricter public declarations introduced by rc2 remain intentional:
`SpectralNumerics.spectral_storage` and the three ambiguous `Output` policies
must be written explicitly.  PSSolver-Control migrated its five affected
call sites without changing canonical simulation metadata.

The independent planar Control adapter also updated its constitutive source
pin after Provider commit `4ca7a004c394135290285d60a4989cd5a1446240`
recorded that its six consumed Beris--Edwards helper definitions are
byte-identical between the old and rc3 Provider commits.  That evidence is
symbol-scoped and does not claim whole-file equivalence.

## Independent Consumer state

PSSolver-Control commit `03dbc39975978d9d0b7f84965015da24f9d3c206`
passed its source and local installed-wheel CPU suites.  It additionally uses
a resolution-aware Channel Taylor adjudicator that excludes remainder levels
below the measured numerical resolution floor while retaining central finite
differences and explicit wrong-gradient negative controls.

PSSolver-Control remains an independent distribution.  This Provider release
record does not bundle its code or authorize an optimizer campaign or a
scientific control conclusion.

## Release boundary

The final rc3 source distribution and wheel must be built from the frozen
release-metadata commit and pass an isolated installed-wheel CPU smoke.  A
single H100 job must then verify the final installed Provider/Consumer wheels,
the CUDA-only tests, and real CUDA Channel functional gradient/Taylor paths.
Only after those gates pass may the annotated `v0.2.0rc3` tag and GitHub
prerelease be created.

No production default, checkpoint schema, functional API version, model
equation, or `nematics3d` installation is changed by the version update.
