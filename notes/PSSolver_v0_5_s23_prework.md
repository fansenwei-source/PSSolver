---
stage: S-23
status: local_prework_not_reviewed
basis: 64fc24dcd59cbafef105062d33685dd4fe6b2d65
release_reference: 4c12228aba01e8483e80d28a1471ab72384d5fda
decision: D-0044
---
# S-23 local M0 prework

Architecture: sections 6 (qualification outside the package) and 7 (verification).
The one-time D-0044 exception changes start order only. S-22 remains frozen at
the basis above and is reviewed first. No production changes, GPU, HPCC or push.

## Execution specification, before numerical measurements

- Use two new system-site-packages venvs based on the same read-only Nematics3D
  interpreter; install the published 0.4.0 wheel and the S-22 candidate wheel
  with `--no-deps`. Validate import origins and distribution hashes.
- Four CPU threads, CUDA hidden. Each release replay and candidate uses the same
  absolute computation path sequentially, with separate initially empty compiler
  caches. Output records are outside that reusable computation path.
- Expand design section 6.1 without reducing grids, steps or restart points.
  Validate each option with the installed reference; rejected combinations are
  recorded with exception type and message, not silently omitted. The explicit
  Plane linear-space × summation-space pair is a Cartesian product. Other axes
  are varied about defaults, with both Plane runtimes and both dtypes.
- Hash contiguous raw tensor bytes, including nonfinite bit patterns; store
  shape/dtype separately. Compare reference replay 1/2, then candidate only with
  identical numerical fingerprints. A mismatch is not excused by tolerance.
- Track evolved fields, constraint fields, progress and persistent solver state.
  Checkpoint synchronization and observation are explicit trace phases, not
  hidden side effects of hashing. Include continuous and split trajectories.
- Newly generated fixtures are authentic outputs of the installed 0.4.0 writer.
  Historical format versions come from the existing immutable release corpus;
  do not relabel current output as an older format. Future S-24/S-25 rejection
  dispositions are inventory only here: implementing them would change runtime.
- MMS derivation runs in the dedicated SymPy environment, independently of the
  package. Predictions: resolved trigonometric modes have roundoff-level spatial
  residuals; smooth periodic truncation converges spectrally; implicit Euler has
  first-order global error. Existing numerical tolerances are not relaxed.
  Generated expressions and their hash are recorded before numerical checks.
- S-19 H100 evidence is referenced only; CPU work supplies no GPU qualification.

This is not a claim that the matrix, fixtures or MMS gates have passed. Results,
failures and remaining work are reported separately; S-24 is inventory only.
