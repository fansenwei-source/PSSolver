# Phase 4 P4.5 combined SBDF2/modal-block canary

Status: `P4_5_LOCAL_COMPLETE_OPT_IN_REFERENCE`.

P4.5 combines the constant-step SBDF2 contract with the qualified P4.4
two-component modal block in one opt-in reference workflow.  The analytic
problem uses the P4.4 constant-coefficient reaction--diffusion block as the
implicit operator and a nonzero scalar identity-growth term as the explicit
operator.  The exact solution remains a matrix exponential multiplied by the
commuting scalar exponential.

The canary performs one projected semi-implicit Euler startup step and then
uses the frozen SBDF2 formula.  Its four-level `0.04, 0.02, 0.01, 0.005`
temporal ladder at final time `0.4` produced L2 errors
`2.6003498227867753e-4`, `6.42568019216997e-5`,
`1.5976331181916127e-5`, and `3.983485757976372e-6`.  The corresponding
orders are `2.0167846364227993`, `2.0079130346743805`, and
`2.0038328225807054`, all above the frozen `1.8` threshold.

The fixed execution workspace owns the current explicit RHS, assembled
implicit RHS, modal output, and four modal scratch tensors: seven tensors in
total.  Persistent history owns distinct copies of the previous evolved
spectrum and previous explicit RHS.  History is committed only after a
successful inverse transform, optional spectral refresh, and progress commit.
Injected stage failures preserve the caller's last complete state and history.

The P4.5 checkpoint uses the already frozen
`provisional_generic_multistep_v1` format identity with an additional exact
workflow identity.  It records stepper/model/operator metadata, complete
runtime and refresh clocks, tensor shape/dtype/finite/SHA-256 records, and both
history tensors.  Continuous execution, in-memory split execution, and a
checkpoint/restart split are byte-identical on CPU.  Missing COMPLETE,
tampered history, non-finite data with rewritten hashes, and changed `dt` are
rejected before advancing.

The local CUDA smoke also passed on an NVIDIA GeForce RTX 3060 Ti and verifies
that a logical `cuda` request is rebound to the concrete allocated device
identity.  This is an implementation smoke only; it does not replace the H100
closure required by P4.6.

The composition lives in `pssolver.experimental`, which may depend on both the
integrator and operator layers without introducing a reverse dependency from
`integrators` to `operators`.  It remains direct-import-only, is not exported
from the package root, does not connect to RuntimeState, StepProgram, Plane, or
the production checkpoint, and changes no default.
P4.5 local completion authorizes P4.6 closure qualification; it does not
complete Phase 4 or authorize Phase 5.

The machine-readable record is
[phase_4_p45_combined_sbdf2_modal_block.json](phase_4_p45_combined_sbdf2_modal_block.json).
