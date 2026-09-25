# P7.7.12 H100 evidence recovery

Status: `READY_CPU_ONLY_EVIDENCE_RECOVERY`.

The single P7.7.12 H100 job produced all 36 reports and finite scientific
outputs. Direct and public entries produced byte-identical final Q, velocity,
pressure, and same-runtime resumed artifacts. The original analyzer stopped on
two qualification-contract mistakes rather than on a solver, CUDA, numerical,
or scientific failure.

## Performance attribution

`run_simulation()` selects and invokes one existing application runner, then
wraps the returned application result. It contains no timestep loop. The
application workflow starts and stops its own `elapsed_seconds` timer inside
that invocation, and `_public_result()` passes this elapsed value through
unchanged. Public dispatch and result wrapping therefore occur outside the
timer used by the original direct/public ratio.

Consequently, ratios from separate 60-step application processes cannot
measure public-wrapper overhead. The two high Channel ratios coexist with
byte-identical final state, identical transform/runtime selection, unchanged
peak memory, no fallback, and a third ratio near unity. They are retained as
diagnostic timing observations, but are not a valid causal gate on the public
wrapper. P7.7.6 remains the authoritative qualification of the underlying
production application/runtime performance.

The recovery analyzer does not simply widen the old 1.05 threshold. It replaces
the misattributed measurement with a static no-public-timestep-dispatch audit,
exact application-timer passthrough, exact direct/public scientific artifact
identity, the original memory limits, and the frozen P7.7.6 runtime evidence.

## Restart diagnostics

A workflow checkpoint restores numerical runtime state and its completion
clock. It does not claim to serialize the observation history already written
by an uninterrupted run. A continuous run can therefore contain an initial
diagnostic record that a resume-only output cannot contain, even when both
reach an identical final physical state.

The recovery preserves the strict byte-identity requirement for final Q,
velocity, pressure, and `COMPLETE`. Direct and public resume outputs must still
be entirely byte-identical. For continuous versus resumed diagnostics, it
loads the immutable output files and requires the diagnostic record at the
common final physical step to be byte-identical in both NPY and CSV. It does
not require unequal observation histories to have equal whole-file hashes.

## Execution boundary

Recovery is analysis-only and CPU-only. It reads the immutable 36-report H100
matrix and its existing output directories, verifies the original checksum
manifest, and writes a new independent recovery archive. It must not rerun a
simulation, submit a GPU job, modify the old archive, change the solver,
promote a runtime, or authorize Phase 8 or Phase 9. A PASS makes P7.7.12 and
P7.7 complete and establishes eligibility for separate Phase 8 planning only.
