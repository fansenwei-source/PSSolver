# PSSolver v0.2.1 local package preparation

Date: 2026-10-20

Classification: `PASS_V0_2_1_LOCAL_PACKAGE_PREPARATION`

Release source `ca9c6729670f07639e7877881c777d1653f4a592` passed the
complete CPU suite: 4134 passed, one existing optional `nematics3d` skip,
seven exact CUDA-only deselects, zero failures, and eight passing subtests.

A clean prefixed Git archive was used to build the sdist and wheel with an
isolated PEP 517 build environment.  The release-candidate artifacts are:

- source archive SHA-256:
  `2541432b9c5f8febdff68e913f005e2b1cc223a8234f0f06c87d2bca60cbc585`
- wheel SHA-256:
  `0f0f1c43a920582d3d707d285fe06502ff1852e6cca24f37872b06669201da8a`
- sdist SHA-256:
  `cf795a976494dd552f3db843ba88185f77a0b9510ef3e60d4a9b9582413b6eba`

The wheel was installed with `--no-deps` into a new isolated venv.  `pip
check`, the public version/API checks, source-shadow audit, console help, and
13 installed-wheel N-1 checkpoint tests all passed.  Rebuilding a wheel from
the sdist produced the same 263 member contents; all 258 packaged Python files
were identical between sdist and wheel.

The already completed small-H100 N-1 qualification (Job `10902665`) remains
applicable because the only runtime difference in this release-source commit
is `_version.py: 0.2.0 -> 0.2.1`.

This evidence-only record is not the release source.  Push, annotated tag,
GitHub Release, PyPI publication, and any additional H100 job remain
unauthorized.
