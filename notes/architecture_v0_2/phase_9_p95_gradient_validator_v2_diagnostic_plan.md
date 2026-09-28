# Phase 9 P9.5 recovery: stable gradient-validator v2 diagnostic

## Why a second diagnostic is necessary

The first scale sweep completed on H100 in Job `10845029`.  All six autograd
paths were finite and nonzero, repeated objectives were bitwise deterministic,
and no hidden detach was observed.  Nevertheless, no activity epsilon passed
the existing `2e-5` relative-error reference on both R128 and R320.  The best
minimax epsilon, `3e-3`, still produced a maximum relative error of about
`3.94e-4`.  R320 also failed the frozen state audit at about `3.85e-5`.

Those results rule out selecting a lucky fixed epsilon.  They instead identify
an ill-conditioned validation calculation: the current validator subtracts two
already reduced, nearly equal global scalar objectives.  A small independent
R128 prototype showed that moving the subtraction before the global reduction
improved the state error at `1e-6` from about `9.07e-6` to `5.35e-7`, while the
activity error at `1e-4` improved from about `7.02e-3` to `9.60e-4`.  Thus
scalar cancellation is real, but direction conditioning must also be tested.

The machine-readable contract is
[phase_9_p95_gradient_validator_v2_diagnostic_plan.json](phase_9_p95_gradient_validator_v2_diagnostic_plan.json).

## Analysis-only matrix

One fresh process per grid evaluates R128 and R320 using the immutable P9.5
inputs.  The preceding H100 diagnostic produced bitwise-identical repeated
objectives and identical same-grid trials, so this round spends its budget on
method and direction coverage rather than repeating deterministic evidence.
It does not modify
`pssolver.functional.validation.evaluate_periodic_activity_gradients`.

For state and activity independently, the tool compares three deterministic
directions:

1. the existing frozen oscillatory direction;
2. an independently constructed smooth low-mode direction;
3. a gradient-aligned direction used only to diagnose conditioning.

At every epsilon, it records two algebraically equivalent central differences:

1. `legacy_scalar`, which reduces each objective and then subtracts;
2. `paired_quadratic`, which forms
   `Re(conj(y_plus-y_minus) * (y_plus+y_minus))` pointwise before reduction.

The second expression computes the same quadratic-objective difference but
avoids subtracting two large reduced scalars.  The report also retains
one-sided first-order Taylor remainders and empirical error orders across each
epsilon ladder.

## Interpretation boundary

The independent low-mode direction prevents a successful result from relying
only on an autograd-derived direction.  The gradient-aligned direction is a
conditioning probe, not independent proof of derivative correctness.  The six
separate finite/nonzero path checks remain the hidden-detach gate.

The analyzer may identify cross-grid method/direction/epsilon candidates.  It
cannot change the validator, qualify P9.5, or authorize P9.6.  A later reviewed
source change is required to adopt any validator-v2 contract, followed by CPU
tests and a new complete P9.5 H100 qualification.

The forward and reverse PDE calculations remain the same pseudo-spectral
Beris--Edwards timestep and direct periodic Fourier Stokes solve.  Finite
differences appear only in this independent validation diagnostic.
