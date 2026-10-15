# PSSolver 0.2.0rc5 release candidate

PSSolver 0.2.0rc5 is the compatibility and robustness follow-up to the
immutable `v0.2.0rc4` prerelease.  The rc5 work queue was created from the
merged post-release audit and was executed in order: W0 evidence registration,
W1 documentation alignment, R5-1 through R5-12, and the optional R5-13 hygiene
batch.

## Closed work queue

The release candidate restores real rc1--rc3 functional checkpoint loading,
prevents non-finite workflows from publishing successful completion, rejects
incompatible legacy Channel dynamics before mutation, fixes packaging and
installed provenance, restores exact caller TF32 state, and hardens public
errors, device normalization, checkpoint integrity, pressure postconditions,
cross-version observation behavior, metadata contracts, compiled-simulation
identity, restart validation, and durable no-replace checkpoint publication.

R5-13 deliberately retains three compatibility/deferred boundaries:

- `DEFAULT_SPECTRAL_STORAGE` remains a frozen v0.1 root export;
- legacy functional bridge metadata continues to use the released version-1
  identity while current bridge formats remain version 3;
- lifting remains fail-closed, so effective lifting is not inferred from a
  non-executable runtime.

## Stable public boundary

The package version changes to `0.2.0rc5`.  Functional API version 1.0,
adapter versions, checkpoint schemas, the four qualified model--geometry
combinations, batch-one functional scope, and production defaults do not
change.  Compiled runtimes remain opt-in.  PSSolver-Control remains an
independent consumer, and `nematics3d` remains an unmodified optional analysis
dependency.

## G-1 qualification boundary

This source-freeze preparation is not a G-1 PASS.  The frozen source must still
pass the full source suite, a fresh installed-wheel CPU suite, the real rc1--rc4
checkpoint matrix, any required PSSolver-Control migration and installed-wheel
qualification, and one explicitly authorized single-H100 cumulative run.  The
H100 run must cover CPU↔CUDA checkpoint compatibility, TF32 caller state
including `"medium"`, standard numerical non-regression, memory, and
performance gates.

No push, tag, GitHub release, PyPI publication, default promotion, optimizer
campaign, or scientific-control conclusion is authorized by this record.
