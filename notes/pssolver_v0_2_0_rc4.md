# PSSolver 0.2.0rc4 release candidate

PSSolver 0.2.0rc4 closes the five correctness and robustness batches opened
after `v0.2.0rc3`.  It does not expand the public capability catalog, change
functional protocol version 1.0, or promote an opt-in runtime to a production
default.

## Correctness closure

RC4.1 unifies Plane periodic Nyquist handling across full-complex and
Hermitian-half storage.  RC4.2 hardens checkpoint identity and transactional
publication.  RC4.3 makes the supported lifting boundary explicit and carries
its identity through fresh construction and restart.  RC4.4 freezes the
Channel pressure contract and its public diagnostics.  RC4.5 closes public
validation, cross-runtime robustness, scoped TF32 ownership, independent
Consumer compatibility, and cumulative H100 non-regression.

The final RC4.5.8 record is
`notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json`.  It preserves the failed
external-harness layers, binds the final CPU-only recovery, and records 96/96
profiles and 16/16 runtime groups passing without another H100 submission.

## Compatibility boundary

The package version changes to `0.2.0rc4`; the stable functional API remains
`1.0`.  Existing production defaults, functional batch-one scope, checkpoint
compatibility readers, qualified model--geometry combinations, and
PSSolver-Control's status as an independent distribution remain unchanged.
`nematics3d` remains an optional analysis dependency and is neither modified
nor bundled.

## Release boundary

This is the source-preparation record.  Before an annotated `v0.2.0rc4` tag
or GitHub prerelease is created, the frozen source must produce a clean source
distribution and wheel, pass an isolated installed-wheel CPU smoke, and pass
one separately authorized installed-wheel H100 release smoke.  No automatic
merge, default promotion, package publication, optimizer campaign, or
scientific control conclusion is authorized by this record.
