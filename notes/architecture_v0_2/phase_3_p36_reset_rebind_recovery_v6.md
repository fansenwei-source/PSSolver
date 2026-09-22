# Phase 3 P3.6 reset/rebind recovery v6

Status: `P3_6_RUNTIME_APPLICABILITY_NEGATIVE_CONTROL_RECOVERY_PENDING`.

Recovery v6 completed the relative-L2 schema correction.  Strict JSON now
represents a zero-reference, nonzero-difference relative norm with JSON
`null` and an explicit status, while the finite elementwise violation count
remains authoritative.  Legacy production passed all three CPU trials under
the strict `128 * eps` contract, and its deliberately unsynchronized control
produced 9,417 violating elements.

The run stopped because the same numerical-discrimination requirement was
applied to the separated canary.  Omitting the explicit observation sync in
that runtime produced no mixed-tolerance violation.  This agrees with the
earlier local diagnostic: the legacy path has an observable static-field
dependency at that point, whereas the separated path's algebraic execution
does not expose the same stale-state behavior.

A negative control can establish sensitivity only where the tested dependency
is present.  The next recovery must keep the legacy numerical-discrimination
gate, but treat the canary control as diagnostic.  For the canary, the formal
gate is instead exact call-order instrumentation plus successful production-
order evolution under the strict float64 tolerance.  Lack of an error after
intentionally omitting a redundant operation is not a solver failure.

No H100 job was submitted.  The next recovery retains the single authorized
H100 submission after completing the remaining canary CPU trials.

The machine-readable record is
[phase_3_p36_reset_rebind_recovery_v6.json](phase_3_p36_reset_rebind_recovery_v6.json).
