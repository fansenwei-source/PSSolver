# Phase 3 P3.6 reset/rebind recovery v5

Status: `P3_6_RELATIVE_L2_SERIALIZATION_RECOVERY_PENDING`.

Recovery v5 implemented the strict `128 * eps` mixed float64 contract and
passed its tolerance synthetic tests.  It stopped before a formal CPU trial
or any H100 submission because a deliberately unsynchronized negative control
contained a zero-norm reference tensor and a nonzero difference.  The
mathematical relative L2 value is therefore infinite, and the strict JSON
writer correctly rejected Python's non-standard `Infinity` token.

This is a result-schema problem, not a numerical or solver failure.  Relative
L2 is diagnostic only; the pre-registered elementwise mixed tolerance and its
violation count remain the qualification authority.  The corrected schema
must record a JSON `null` relative-L2 value plus an explicit zero-reference
status, while preserving finite numerator, denominator, Linf, RMS, allowed
error, and violation counts.  It must never replace infinity with an arbitrary
large finite sentinel or enable non-standard JSON values.

No H100 job was submitted, so the next recovery may complete the CPU gate and
retain the single authorized H100 submission for reset/rebind and the eleven
negative gates.

The machine-readable record is
[phase_3_p36_reset_rebind_recovery_v5.json](phase_3_p36_reset_rebind_recovery_v5.json).
