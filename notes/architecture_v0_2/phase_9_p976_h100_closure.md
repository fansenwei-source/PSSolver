# Phase 9 P9.7.6: independent Channel consumer closure

## Result

P9.7 is complete with classification
`PASS_P9_7_6_INDEPENDENT_CHANNEL_CONSUMER_INSTALLED_WHEEL_H100`.  The
machine-readable authority is
[phase_9_p976_h100_closure.json](phase_9_p976_h100_closure.json).  It binds
provider commit `3b6fb9c71d3cd68a72caf3b184bdf0a61826e39c`, independently
maintained consumer commit `73c022b00fff71b964b91a4d497419fff21e34ef`, and the
final recovery archive.

The provider and consumer source suites passed, both wheels were installed in
an isolated environment, and the installed-wheel CPU audit passed without
source shadowing.  The only H100 job completed successfully.  It qualified the
public batch-one Channel functional runtime, production/functional identity,
bitwise replay, checkpoint-stride invariance, durable restart, the custom
implicit pressure adjoint, finite-difference and Taylor gradients, bounded
adjoint memory, and all eight negative gates.

## Numerical evidence

Production and functional velocity, pressure, physical Q, and spectral Q were
byte-for-byte identical for the frozen one-step comparison.  Checkpointed and
full-history gradients had relative L2 error zero.  The maximum three-direction
central finite-difference relative error was
`3.13844556216355e-06`, below the frozen `2.0e-05` bound.  The three
gradient-subtracted Taylor rates were `2.112358347151925`,
`2.121849620746324`, and `1.9712815120499314`.

The implicit pressure adjoint retained no iteration tensors and its graph size
did not grow between the two- and five-iteration checks.  This is a small-grid
qualification result, not a production-scale speed claim.

## Pressure-iteration interpretation

The qualified Channel problem requested twelve pressure iterations.  The last
recorded primal and transpose relative residuals were approximately
`3.50e-07` and `7.05e-07`, respectively, while the configured relative
tolerance was `1.0e-10`.  This is not a contradiction in the current runtime:
when a fixed iteration count is supplied, the count is the deterministic work
cap and the tolerance is an early-convergence criterion, not an unconditional
postcondition after the cap is exhausted.  P9.8 must expose this distinction
unambiguously in stable metadata and validation semantics.

## Archive note

The first finalization attempt incorrectly included a live log in its checksum
manifest.  The invalid manifest and traceback remain preserved.  Finalization
was repeated without rerunning any scientific command, and the authoritative
manifest verifies all 54 entries.  This was an archive-harness defect, not a
solver or scientific failure.

## Authorization boundary

This closure completes P9.7 and authorizes P9.8 planning.  It does not itself
stabilize the provisional functional API, authorize batch sizes above one,
change a production default, run an optimizer or scientific-control campaign,
or complete Phase 9.  P9.8 implementation and its final installed-wheel H100
closure remain separately gated.
