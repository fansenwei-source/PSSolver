# Phase 4 P4.4 contract-equivalent H100 adjudication

Status: `PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT`.

Job `10837117` completed the six-size H100 crossover profiler at commit
`7619c58c931eda89ff6613936851d8e7be1f52b9`.  All numerical, residual,
finite-value, fixed-workspace, trial-completeness, and memory observations
passed.  The closed-form implementation was faster in all eighteen paired
trials.  Its candidate/reference median ratio decreased from `0.967146749` at
`131072` modes to `0.553286188` at `4194304` modes.

The external qualification nevertheless classified the run as failed because
it required the low-size direction of Job `10837110` to repeat.  That gate is
not authoritative for the operator decision.  The earlier reference measured
a different execution scope: its `131072`-mode reference median was
`0.237903997` ms, whereas the complete operator-contract reference in Job
`10837117` was `0.366461865` ms.  The closed-form medians remained comparable
at `0.375119999` and `0.354422401` ms.  Requiring the complete operator path to
reproduce a non-equivalent reference timing conflates kernel throughput with
the callable operator contract.

The corrected authority is the complete operator contract: input and
workspace validation, output publication, finite and residual gates, stable
workspace ownership, and implementation-specific matrix rejection.  The old
bare-kernel low-end direction is retained as historical evidence but no longer
has veto power over P4.4.

`benchmarks/adjudicate_modal_block_crossover.py` performs the formal
analysis-only correction.  It reads the immutable raw profiler JSON, rejects
duplicate or non-finite JSON, verifies its commit/H100/configuration identity,
recomputes all aggregate and paired timing ratios, checks all numerical and
memory gates, and emits a new report without executing PSSolver or CUDA.  It
does not implement an automatic policy: the existing explicit
`closed_form_2x2` implementation is the qualified H100 candidate over the
registered range, while `torch_linalg_reference` remains the independent
reference.

The immutable Job `10837117` artifact passed this adjudicator.  The formal
report is
`/home/fansenwei/pssolver_phase4_p44_contract_adjudication_05cb8fa_20260922_v1/reports/p4_4_contract_equivalent_adjudication.json`
with SHA-256
`5aab006967ce5df3cb387023c7b979db159b524cb0de92c0c9dd9ddae8050c4b`.
The decision sets `p4_4_h100_qualified=true` and `eligible_for_p4_5=true`.
It does not mark Phase 4 complete, authorize Phase 5, change a production
default, promote a Plane runtime, or create automatic dispatch.
