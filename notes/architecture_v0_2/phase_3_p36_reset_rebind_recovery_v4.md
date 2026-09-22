# Phase 3 P3.6 reset/rebind recovery v4

Status: `P3_6_RESET_REBIND_ROUNDOFF_CONTRACT_ADJUDICATION_PENDING`.

Recovery v4 corrected the production restoration order and passed every
initial-state isolation, immediate state/storage rebind, workspace, progress,
and representation-ledger gate on CPU.  Its deliberate negative control
confirmed that omitting static synchronization causes errors between
approximately `1e-8` and `1e-4`, depending on the field.

With the production restore order, the remaining legacy differences fell to
float64 roundoff: the largest physical difference was
`1.3877787807814457e-17`, the largest spectral difference was
`4.577566798522237e-16`, and the Q/u/p differences were at most approximately
`1e-17`.  The run stopped only because the external qualification contract
required byte identity and exact zero error for a reset-versus-independent-
construction comparison.

That requirement is stronger than the frozen reset/rebind contract.  The
qualified reset API promises state replacement, adoption of current solver
field storage, workspace and progress semantics, and valid subsequent
evolution.  Byte identity remains required for the already-qualified
continuous and checkpoint/restart workflow trajectories, but independent
reconstruction can legitimately differ at roundoff through algebraic
recomputation order.  The next recovery must pre-register a strict mixed
float64 tolerance for this diagnostic while leaving all workflow byte-
identity oracles unchanged.

The machine-readable record is
[phase_3_p36_reset_rebind_recovery_v4.json](phase_3_p36_reset_rebind_recovery_v4.json).
