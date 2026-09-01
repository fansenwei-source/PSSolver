# Shendruk reproduction decisions

Last updated: 2026-09-01

This note records modeling and validation decisions for the current
Shendruk-reproduction branch. It does not assert that the reproduction has
already been validated.

- Issue 1 (inertial momentum): deferred. First attempt the reproduction with
  the current quasistatic Stokes model. Revisit inertia only if the benchmark
  shows a systematic discrepancy that cannot be explained by numerical
  convergence, initialization, or the other documented modeling choices.
- Issue 2 (tangential zero mode) and issue 3 (kinematic free slip): retain the
  current choices for the first attempt and revisit them if reproduction fails.
- Issues 4 and 5 (anchoring branches and additional elastic constants): later
  extensions; no current implementation change.
- Issue 6 (different numerical method and initialization): current validation
  priority. Validate time step, grid, dealiasing, defect-core resolution, and
  initialization sensitivity before a full activity-number scan. The
  same-grid defect-core analyzer now supports a strict cubic-half-reference
  versus two-thirds sensitivity comparison, and the validation runner has an
  analysis-only `core_dealias` stage that can reuse the four completed R320/R512
  runs without permission to launch simulations. The HPCC core-dealias analysis
  and its scientific interpretation are still pending; implementation alone
  does not close issue 6.
- Issue 7 (remaining equation-level tests): still open and independent of
  issue 6; completing convergence studies does not close it.
- Issue 8 (shared Beris--Edwards implementation): resolved for the current
  script.
- Issue 9 (isotropic stress/effective pressure): explanation only; no repair is
  required.
