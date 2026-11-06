---
stage: S-23
status: environment_accepted_numerical_prework_partial
decision: D-0045
supersedes: "PSSolver_v0_5_s23_environment_hold.md: environment block only"
---
# S-23 isolated dependency recovery and resumed CPU draft

Architecture: sections 6 and 7. No layer moved and no `pssolver/` file changed.
S-22 remains `64fc24dcd59cbafef105062d33685dd4fe6b2d65`; its wheel is unchanged.
The earlier hold record and v1 archive are retained, not rewritten.

## Environment gate

Evidence: `/home/fansenwei/pssolver_v0_5_s23_baselines_20261106_v2`.
Original Nematics3D and daily protection/configuration: 51,215 inventory entries,
identical before clone, after clone, after installation and after the CPU runs.
The combined manifest SHA-256 is
`f0361e9754b698ebf7f9e738a824db80481a8b0bf9ec0a5864028dc7bdf19031`.

The real conda clone used `--copy --offline`; optional automatic `.pyc`
compilation was deferred in that invocation only. A child-process audit rejected
premature source/target interpreter execution. No installed conda files changed.
Before first cloned-Python startup, eight cloned guard/editable entries and their
caches were moved to a quarantine outside the dependency prefix. In particular,
the guard symlink itself was moved, not its original target. No remaining clone
symlink points outside the copy; no regular clone file shares its original inode.

The dependency copy contains 13,353 byte-identical numerical package/library
files from NumPy, SciPy, Torch, NumExpr and their bundled NVIDIA dependencies.
Its full frozen manifest is
`d2dbb5faafc666d0414e2985c016aa8ca976e6033e5d9fc292ec9d623089ffdc`, unchanged
after installation and CPU tests. No extra dependencies were installed.
Two new `--system-site-packages --without-pip` venvs use this copy; pip is invoked
from the shared copy with each venv's `python -I -B -m pip`. Only the two specified
Provider wheels were installed, `--no-deps --no-index --no-cache-dir --no-compile`.

Both environments pass pip check, startup-stream, prefix, path, hook, package
origin, wheel-content and RECORD checks. The 114 actual loaded binary files have
identical paths and hashes across the two processes; OS libc/libstdc++/driver
libraries are explicitly listed separately from copied numerical libraries.
The CUDA-enabled Torch build was retained; CUDA was hidden and not initialized.
Four numerical threads and one Torch interop thread were used. Optional PyYAML
formatting warnings are retained; they are not `.pth`/guard errors.

Final environment gate `environment/acceptance_v2.json`:
`8e00636107683677a659f2e8fcb40ff06d1349ae47a45a13141f977bccf5170d`.
The first aggregate checker omitted the wheel-owned top-level CLI module from
its expected-name list. Its failed output remains in `acceptance.json`; v2 derives
the exact list from the approved wheel. Neither environment was changed to pass.

## Resumed CPU checks, not complete S-23 qualification

- The private `_solver` draft error is covered by actual stable functional
  runtime construction, not a mock. Both installed wheels: six focused tests pass.
- A separate draft bug wrote float64 initial snapshots for float32 scenes. The
  harness now writes the declared dtype. The first option probes and the first
  focused test's incorrect assumption that functional Channel accepts float32
  are retained as superseded draft evidence; neither becomes qualification data.
  The corrected test checks the actual v0.4 rejection, with no skip/xfail.
- Corrected draft option probe: 99/99 raw records identical, 85 accepted and
  14 rejected, including exception type/text and identity. Comparison SHA-256:
  `210d7eb88937f6d36cc9c1d5e34f86be372d542aed846d3b7231bf5405fecd1b`.
- Release replay 1, replay 2, then candidate, sequentially reuse the same exact
  computation path. Compiler cache paths are separate and initially absent.
  All 99 raw result JSON files match byte for byte, with identical fingerprints.
  Each run: 5,365 continuous rows, 5,410 restart rows, 332,738 tensor hashes,
  zero nonfinite tensor records. Comparison SHA-256:
  `01767086642e192aeb754cc0d736b3c5d8407832f803786cea5805275e0a386c`.
- These comparisons preserve 135 continuous/restart phase bookkeeping differences
  already present in the reference: 133 involve `constraints_current`, three
  involve pressure/velocity fields at a later checkpoint of unobserved runs
  (one row overlaps). They are not normalized or asserted equal. Sampling phases
  and persistent solver-cache coverage still need the full S-23 harness audit.
- `tools/check.sh --base 64fc24d...` passes with an archive-local installed-wheel
  bootstrap (28 quick tests; import contracts; Ruff; 38 unchanged historical mypy
  errors plus one note, no new diagnostics). A preceding PYTHONSAFEPATH attempt
  failed on the tool's `_source` sibling import and is retained. The bootstrap
  preloads the installed Provider before adding source-only tool/test paths.

Remaining: complete the option combinations, VJP, authentic checkpoint corpus
and rejection inventory, compiler probe, independent MMS, baseline prerequisite
category, reachable-clone full suites and final local-evidence verification.
No R request or S-23 qualification completion is asserted by this prework record.
The local hub check reports zero errors and one pending: R-0071 still awaits
Claude, consistently with D-0044's temporary prework exception.

## Forward source bindings and engineering report

New source hashes, without rewriting the historical S-22 record:

| File | SHA-256 |
|---|---|
| `qualification/identity/runtime_case.py` | `94941a4852631c8d2bba92c7cf7db3376b5ad08cd5496bd738802d89863209ac` |
| `tests/test_v0_5_s23_identity_scaffold.py` | `8e11080fa4334f34765c235ec999432bb677bf7e8ad186dedef7b007f3495763` |
| `tests/categories.toml` | `a877430bceab56e29cb4438ec1d9ff01190951633064ac4af7a8f04076703fb6` |

No production parallel path was added. Temporary dependency/venv copies are
qualification-only, retained for completing S-23; later pruning needs a truthful
manifest/PRUNED record. Existing atomic artifact I/O is reused. Package delta is
zero; qualification delta +1/-1, tests +40/-0 including the category entry.
The entry point is `qualification/identity/trajectory_hashes.py`, whose
`RuntimeCase` calls the installed public/runtime APIs; mathematics remains in
the unchanged wheel. The four Python engineering skills guided the small input
fix and regression tests; no shared mathematical kernel or tolerance changed.
