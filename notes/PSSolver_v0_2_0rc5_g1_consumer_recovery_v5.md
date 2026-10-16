# PSSolver rc5 G-1 Consumer recovery v5

The recovery-v4 H100 job completed the node-local checkpoint preflight, the
environment, CUDA-only, TF32-scope, checkpoint, numerical, and 96-profile
execution stages before stopping at the Consumer gate.

Two independent issues were persisted.  The Consumer's explicit rc4 Periodic
identity pin was rejected only after Python 3.10's runtime-checkable protocol
probe had accessed `checkpoint_bridge`.  Afterwards the external qualification
helper imported `PSSolverIntegrationUnavailable` from the nonexistent
`pssolver_control.exceptions` module instead of `pssolver_control.errors`.

The Consumer fix validates an explicit public identity pin before the protocol
probe.  The Provider qualification helper now imports the actual installed
error module and has an AST regression test.  Neither change modifies the
PSSolver runtime package, numerical implementation, scientific thresholds,
control algorithm, functional API version, adapter version, or production
default.

The historical G-1 stage-producer and HPCC-plan records remain byte-for-byte
unchanged.  Their two changed files are registered as explicit successors in
`PSSolver_v0_2_0rc5_g1_consumer_recovery_v5.json`.

Local Consumer CPU validation passed with 337 tests and the six pre-existing
CUDA/benchmark skips.  The Provider full CPU suite passed 3,724 tests and eight
subtests with the seven frozen CUDA-only nodes deselected.  This record does
not authorize a push or H100 job and does not claim G-1 qualification
completion.
