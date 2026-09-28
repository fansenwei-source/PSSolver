# Phase 9 P9.5: stable gradient-validator v2 source contract

## Purpose

The analysis-only H100 diagnostic at commit `f592aef` and Job `10845746`
found a stable, independent, cross-grid directional-gradient contract.  This
source change adopts that contract in the formal periodic functional validator;
it does not change the Beris--Edwards equations, pseudo-spectral timestep,
Stokes solve, public control API, or production default.

The machine-readable contract is
[phase_9_p95_gradient_validator_v2_source_plan.json](phase_9_p95_gradient_validator_v2_source_plan.json).

## Frozen validator-v2 calculation

The formal float64 audit retains the same fixed quadratic objective and all six
finite/nonzero autograd-path checks.  It changes only the independent
directional finite-difference calculation:

1. state and activity use deterministic smooth low-mode directions that are
   independent of autograd;
2. the state epsilon is `1e-5`;
3. the activity epsilon is `1e-2`;
4. the relative-error tolerance remains `2e-5`;
5. the quadratic central difference forms
   `Re(conj(y_plus-y_minus) * (y_plus+y_minus))` before global reduction;
6. the report records the direction kind, finite-difference method, epsilon,
   direction cosine, errors, tolerance, and pass status.

The paired formula is algebraically equivalent to subtracting the two fixed
quadratic objectives after reduction.  It is used only by this qualification
audit and does not replace the pseudo-spectral solver with a finite-difference
method.

## Evidence boundary

The diagnostic found maximum cross-grid relative errors of approximately
`6.74e-8` for the state audit and `3.76e-8` for the activity audit.  The
corresponding low-mode direction cosines were positive on both grids.  These
measurements justify the selected contract but do not by themselves close
P9.5.

P9.5 requires a new complete 18-profile H100 matrix: two grids, three roles,
and three trials.  Its analyzer must reject any report that does not identify
validator version 2, `low_mode`, `paired_quadratic`, the frozen epsilons, the
unchanged tolerance, and all six passing gradient paths.  Until that run
passes, P9.5 remains incomplete and P9.6 remains unauthorized.

## Nonclaims

This source step does not qualify larger batches, promote a runtime, change a
production default, implement an optimization algorithm, or claim completion
of P9.5 or P9.6.
