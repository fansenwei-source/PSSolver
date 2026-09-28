# Phase 9 P9.5 recovery: activity-gradient epsilon diagnostic

## Why this diagnostic is separate

The first formal P9.5 H100 job, `10844957`, stopped before VJP timing because
the frozen periodic activity gradient validator failed on R128.  Its two
completed forward profiles were finite, production and functional state hashes
were exact, and R12 passed.  The exception did not persist the validator's
attached report, so it could not distinguish a missing gradient path from a
finite-difference scale failure.

A local R128 reproduction on a CUDA device showed that all six gradient paths
were finite and nonzero and that the state directional derivative passed.  The
activity check failed at the frozen `1e-4` epsilon with relative error about
`7.02e-3`.  Separate fresh processes placed isolated passing points at different
epsilons (`3e-3` or `8e-3`) even though objectives replayed exactly inside each
process.  This is evidence of subtractive-cancellation sensitivity, but it is
not H100 or R320 evidence and cannot change the contract or justify selecting a
single lucky epsilon.

The machine-readable diagnostic contract is
[phase_9_p95_gradient_diagnostic_plan.json](phase_9_p95_gradient_diagnostic_plan.json).

## Frozen diagnostic

One analysis-only H100 job runs three fresh processes for each of R128 and R320
using the already checksum-verified immutable P9.5 inputs.  Each process first calls
the current frozen validator without raising on failure, then records an
activity finite-difference sweep at epsilons from `1e-4` through `1e-2`.

For every epsilon the archive retains the autograd derivative, both repeated
plus and minus objective values, finite-difference derivative, absolute and
relative errors, finiteness, exact objective replay, and whether the existing
`2e-5` tolerance would pass.  It also retains the six path norms, state check,
memory, device, TF32, Git, and input identities.

The analyzer may identify epsilon candidates that pass in all six fresh
processes or report that no common candidate exists.  Neither classification qualifies P9.5, changes an
epsilon or tolerance, or authorizes P9.6.  A later reviewed source commit must
make any scale-contract correction and pass CPU tests before the complete
18-profile P9.5 matrix is rerun in a new output directory.

## Scientific meaning

The finite-difference sweep validates reverse-mode gradients; it is not the
numerical discretization used by the functional runtime.  The controlled
forward and reverse computations continue to use the qualified pseudo-spectral
Beris--Edwards timestep, Fourier derivatives, dealiasing, and direct periodic
Fourier Stokes solve.  Finite differences appear only in this independent
directional-derivative audit.
