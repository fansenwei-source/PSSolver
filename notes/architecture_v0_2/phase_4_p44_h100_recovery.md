# Phase 4 P4.4 H100 performance recovery

Status: `P4_4_H100_QUALIFIED_P4_5_AUTHORIZED`.

The first P4.4 H100 qualification ran at commit
`ea4b051351f3e5b5dc1d3f51a3ed14dc64736603` in Slurm Job `10837110`.
The CPU suites and CUDA-only tests passed.  Both registered H100 sizes passed
the numerical, residual, finite-value, fixed-workspace, singular-rejection,
and memory gates.  The closed-form solve agreed with the independent
`torch.linalg.solve` result to approximately `1.6e-16` relative L2.

The qualification failed only its small-problem performance gate.  At
`131072` modes, the closed form/reference median ratio was `1.576770476`.  At
`4194304` modes, the ratio was `0.636732467`, so the same implementation was
about `1.57x` faster at the large anchor and used less peak memory at both
anchors.  This establishes a device- and size-dependent crossover; it does
not justify discarding the analytic implementation or selecting a threshold
from only two observations.

`benchmarks/profile_modal_block_crossover.py` is the recovery profiler.  It
scans six increasing modal counts with balanced trial order, complex128,
TF32 disabled, common input/workspace validation, independent residual and
finite-value checks, stable workspace identities, CUDA event timing, and
incremental peak-memory observations.  It reports the first measured count
whose complete larger-size suffix satisfies the registered median-ratio gate.
The output is evidence only: it cannot change an implementation, default,
P4.4 qualification, or P4.5 authorization.

The registered H100 recovery was one crossover scan at counts `131072`, `262144`,
`524288`, `1048576`, `2097152`, and `4194304`.  Only after that artifact is
complete may the result authorize a later execution-policy decision; the scan
itself cannot change an implementation or default.

P4.5 remains blocked.  Production defaults, Plane, runtime state, and the
StepProgram remain unchanged.

Job `10837117` subsequently completed this scan.  The closed-form path was
faster in all eighteen paired trials, including the smallest registered case,
while all correctness and memory gates passed.  Its external validator stopped
because the low-size direction of the earlier non-equivalent reference scope
did not repeat.  The corrected analysis-only contract and adjudicator are
recorded in the
[contract-equivalent adjudication](phase_4_p44_contract_equivalent_adjudication.md).
The analysis-only adjudication subsequently passed with classification
`PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT`.  No automatic policy was
required or authorized; P4.5 was authorized on the explicit
`closed_form_2x2` path.
