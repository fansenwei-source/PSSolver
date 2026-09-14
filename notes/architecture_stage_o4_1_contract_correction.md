# Architecture Stage O.4.1: prospective qualification correction

## Why O.4 remains `B_neutral`

Stage O.4 established numerical equivalence through 100 steps, exact
same-backend restart, exact initial-Q identity, valid lifecycle accounting,
and H100 timing parity between `legacy_production` and `separated_canary`.
Its original classification remains immutable.  Two frozen gates failed:
cross-runtime transform-count identity and a single aggregate peak-allocated
memory ratio at the R128 diagnostic scale.

The count gate was inherited from comparisons between two implementations of
the same shadow execution plan.  It is not a valid invariant between the
legacy eager-materialization plan and the separated algebraic-lifecycle plan:
the latter was designed to remove unnecessary transforms.  The aggregate
memory counter also combined the timestep and observation/materialization
windows, while the small R128 denominator amplified fixed runtime overhead.
Neither issue permits retroactively reclassifying O.4.

## Prospectively frozen O.4.1 contract

O.4.1 binds the checksummed O.4 report and collects new profiles under rules
defined before execution.  It does not rerun or reinterpret O.4 science.

Two scales have distinct roles:

- `128 x 128 x 32` diagnoses fixed overhead and is reported but does not set
  the memory or throughput promotion decision;
- `320 x 320 x 80` is the benchmark-relevant decision scale.

Each scale uses three balanced legacy/canary pairs with ten warm-up and twenty
timed steps.  Both profilers expose the same CUDA allocator schema and separate
the timed timestep window from the observation/materialization window.  The
record includes current allocated/reserved bytes at phase boundaries, window
peaks, and retained allocated growth across the post-warm-up timestep window.
Observation materialization growth is not mislabeled as a leak; its total end
state and peak remain explicit R320 ratio gates.

The architecture-aware transform contract requires:

- stable forward and inverse counts within each runtime across all trials;
- canary counts no greater than legacy counts in either direction;
- a strict reduction in at least one direction;
- stable, valid canary lifecycle counters;
- no on-demand physical materialization or retained representation pairs.

The R320 decision retains the conservative `1.03` aggregate mean-timestep and
phase-memory ratios, the `1.05` paired timing limit in at least two of three
trials, and a bounded post-warm-up retained-allocation increase.  Both the
timestep and observation peaks are gated; memory is not excused merely because
the H100 has spare capacity.

## Decision boundary

An `A_recommended` O.4.1 report authorizes only the separate Stage O.5
stability, restart, and promotion decision.  It cannot change
`DEFAULT_PLANE_RUNTIME_PATH`, promote the canary, migrate Channel, or modify
the generic solver.  A `B_neutral` result identifies whether performance,
phase memory, or lifecycle behavior still needs work and retains the legacy
default.
