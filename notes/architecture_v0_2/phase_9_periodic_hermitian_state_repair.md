# Phase 9 periodic Hermitian-state repair

## Status

This repair is developed on an independent branch from Provider commit
`640e93aba6ea81afe590d7f577903dee8c6ffc1c`.  It does not modify the frozen
P9.8.5 archives or the PSSolver-Control source tree.

## Defect

The periodic functional runtime evolves `q_spectral` directly in
`hermitian_half` storage with spectral refresh disabled.  The packed-axis zero
plane, and its Nyquist plane when the physical extent is even, must satisfy

```text
F(k) = conj(F(-k))
```

over all other periodic axes.  Dealiasing removed high modes but did not
project these self-conjugate planes back into the real-field subspace after an
IMEX update.  `irfftn` hid the resulting anti-Hermitian component from the
physical state until that component became large enough to contaminate the
retained dynamics.

The reproduced `32 x 32 x 8` case grew from roundoff with rate
`0.9708108922647739` and became non-finite at `t = 86.48`.

## Repair contract

`BasisAwareSpectralProjector` now provides separate real-spectrum state
projection methods.  They preserve the existing dealias-only `project` and
`project_` contracts while adding the following state invariant:

- select only the packed-axis zero and optional Nyquist planes;
- reflect every other periodic axis by minimum Fourier index identity;
- leave DCT/DST axes unreflected;
- replace each pair by `(F(k) + conj(F(-k))) / 2`;
- cache device index tensors;
- avoid a full-spectrum auxiliary allocation;
- preserve autograd support.

The periodic functional step and production dynamic-state projection both use
the new state projection.  Full-complex Channel paths remain numerical no-ops.
Forward transforms of physical real tensors are not redundantly projected.

The Provider functional identity and periodic runtime metadata record:

```text
hermitian_state_projection = self_conjugate_planes_each_step
```

## Local evidence

- pre-fix reproducer: non-finite at `t = 86.48`;
- repaired reproducer through `t = 100`: violation exactly `0.0` at every
  five-time-unit sample;
- standard CPU long-horizon gate: finite through `t = 200`, with both packed
  self-conjugate planes exactly constrained at every sampled point;
- full CPU suite: `2629 passed, 8 subtests passed`;
- a 120-step cross-version diagnostic changed the physical state by relative
  L2 `4.710330580247838e-17` and Linf `1.6653345369377348e-16`;
- the same diagnostic changed the three one-step VJP components by relative
  L2 values between `6.76e-18` and `4.96e-16`;
- three local CPU timing trials showed no evidence of a regression, but they
  are not a substitute for paired H100 qualification.

## Phase 9 consequence

Existing P9.8.5 results remain valid evidence for their exact frozen Provider
commit.  They are not sufficient as the final long-horizon Periodic closure,
because that commit contains this state-invariant defect.

Before Phase 9 is closed, the repaired Provider must receive a new installed-
wheel qualification covering:

1. the `t >= 200` Hermitian and finite-state gate on CPU and H100;
2. paired H100 forward/VJP performance and memory;
3. Periodic production/functional consistency;
4. checkpoint/restart and negative identity gates;
5. PSSolver-Control full-history/checkpointed gradients, finite differences,
   and Taylor tests;
6. the new canonical Provider identity digest.

The Consumer algorithms do not need to switch away from the pseudospectral
Provider.  Consumer qualification records and any pinned Provider identity do
need to be regenerated against the repaired wheel.
