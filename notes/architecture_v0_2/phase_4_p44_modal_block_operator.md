# Phase 4 P4.4 two-component modal block operator

Status: `P4_4_LOCAL_COMPLETE_GPU_NOT_YET_QUALIFIED`.

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
written entirely with device-resident PyTorch tensor operations.  It can run on
CUDA but has not yet received H100 qualification.

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
format, or default imports them.  P4.5 is deliberately not authorized until
the optimized CUDA path has passed a focused H100 qualification.

The machine-readable record is
[phase_4_p44_modal_block_operator.json](phase_4_p44_modal_block_operator.json).
