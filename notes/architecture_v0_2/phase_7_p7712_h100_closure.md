# Phase 7 P7.7.12: public Simulation API closure

Status:
`PASS_P7_7_12_H100_EVIDENCE_WITH_ATTRIBUTION_CORRECT_RECOVERY`.

P7.7.12 and P7.7 are complete. The original H100 job produced the full
36-report direct/public matrix with finite outputs, no runtime fallback, and
byte-identical scientific artifacts. A subsequent CPU-only recovery reused the
immutable reports and output directories; it launched no simulation and opened
no Q, velocity, or pressure arrays.

The recovery corrected two qualification-contract errors. First,
`run_simulation()` invokes one existing application workflow and wraps its
result outside the workflow's elapsed timer. Independent 60-step workflow
ratios therefore cannot measure public-wrapper overhead. The observed ratios
remain recorded, while P7.7.6 supplies the authoritative underlying runtime
performance evidence. Public and direct scientific artifacts and peak memory
remain identical.

Second, a checkpoint restores numerical state rather than an earlier
observation history. Continuous and resumed runs therefore compare Q,
velocity, pressure, `COMPLETE`, and the diagnostic record at their common final
physical step. All such records were byte-identical. Unequal full diagnostic
histories are not required to have equal file hashes.

The recovered archive verified 55/55 checksum entries and contains a valid
`COMPLETE` marker. The original failed H100 job and its 568-entry archive remain
unchanged; its failure was the old analyzer contract, not a solver or
scientific failure.

No numerical kernel, checkpoint schema, runtime default, or compiled-runtime
promotion changed. Phase 8 planning is eligible, but Phase 8 and Phase 9 remain
unauthorized.

The machine-readable authority is
[phase_7_p7712_h100_closure.json](phase_7_p7712_h100_closure.json).
