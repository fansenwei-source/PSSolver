# Phase 4 P4.4 two-component modal block operator

Status: `P4_4_H100_QUALIFIED_P4_5_AUTHORIZED`.

P4.4 adds a tensor-free declaration for exactly two coupled components and a
device-bound mode-local operator whose coefficient tensor has trailing shape
`(2, 2)`.  The component packing convention is a trailing RHS axis of size
two.  The qualified declaration is limited to constant physical coefficients,
one periodic dimension, mode-independent topology, and mode-dependent
diffusion coefficients.  It is not an arbitrary block algebra API.

Binding validates the component order, geometry and basis capability, real and
spectral dtype pairing, device, coefficient finiteness, and exact block shape.
The CPU reference uses `torch.linalg.solve`.  The optimized implementation uses
the closed-form inverse of a two-by-two matrix over every retained mode and is
written entirely with device-resident PyTorch tensor operations.  Its first
H100 qualification passed every numerical, residual, finite-value, workspace,
singular-rejection, and memory gate, but failed the `131072`-mode performance
gate.  The same path was about `1.57x` faster at `4194304` modes, establishing
a size-dependent crossover rather than a numerical failure.

The optimized path accepts a reusable bounded workspace containing one output
and four mode-shaped scratch tensors.  The workspace and RHS must have the
bound shape, dtype, and device and may not alias.  Singular or numerically
rejected modal matrices fail before the output is written.  Local float64 CPU
tests compare the optimized result with the independent `torch.linalg.solve`
reference and verify the residual.

The qualification model is a two-component constant-coefficient periodic
reaction--diffusion system.  Both off-diagonal coupling entries are nonzero,
and a single cosine mode has an analytic matrix-exponential solution.  Both
solve implementations reproduce the same implicit-Euler modal update.

The declaration, bound operator, workspace, and reference model remain
direct-import-only.  No production runtime, StepProgram, Plane path, checkpoint
format, or default imports them.  The evidence-only crossover profiler and the
recovery plan are recorded in
[phase_4_p44_h100_recovery.md](phase_4_p44_h100_recovery.md).  The completed
six-size recovery showed that the closed-form path is faster at all registered
sizes under the complete operator contract, so an automatic size-aware policy
is no longer proposed.  The analysis-only adjudication documented in
[phase_4_p44_contract_equivalent_adjudication.md](phase_4_p44_contract_equivalent_adjudication.md)
passed with classification
`PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT` and authorized P4.5.

The machine-readable record is
[phase_4_p44_modal_block_operator.json](phase_4_p44_modal_block_operator.json).
