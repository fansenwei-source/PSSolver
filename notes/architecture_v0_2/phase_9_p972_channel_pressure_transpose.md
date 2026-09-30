# Phase 9 P9.7.2: explicit Channel pressure transpose

## Result

P9.7.2 adds an explicit, operator-level transpose contract for the no-slip
Channel pressure Schur complement.  The machine-readable authority is
[phase_9_p972_channel_pressure_transpose.json](phase_9_p972_channel_pressure_transpose.json).

The implementation is an additive adapter in `pssolver.functional`.  The
Phase 7 production file `pssolver/linear_solvers/stokes/channel_no_slip.py`
remains byte-for-byte unchanged, as do the production pressure PCG, Channel
runtime, warm-start policy, runtime defaults, and all existing simulation
entry points.  The independent PSSolver-Control repository is not modified.

## Discrete operator and transpose

In the native periodic-x/Neumann-y/Neumann-z pressure coefficient space, the
frozen production pressure operator is

```text
S = P (-D A^{-1} G) P,
```

where `P` removes the pressure zero mode, `G` maps pressure coefficients to
the three periodic-x/Dirichlet-y/Dirichlet-z velocity-gradient spaces, `A` is
the diagonal velocity Helmholtz operator, and `D` maps the three velocity
components back to pressure divergence.

`ChannelPressureTransposeOperator.apply_pressure_operator_transpose()` does
not call the primal action.  It reverses this exact dataflow and applies the
conjugate transpose of every stored FFT multiplier and dense DCT/DST coupling
matrix:

```text
S* = P G* (A^{-1})* (-D*) P.
```

The declared inner product is the native complex Euclidean coefficient-space
inner product.  Tests exercise arbitrary complex coefficients rather than
only spectra obtained from real fields.  They establish linearity, zero-mode
gauge invariance, the adjoint identity separately for each gradient and
divergence component, and the full identity

```text
<S p, q> = <p, S* q>.
```

Only after that independent construction and dot-product test do the tests
record that the present orthonormal discretization is Hermitian to roundoff.
No implementation path assumes `S* = S` merely from that observation.

## Transpose solve and state isolation

The adapter also provides a zero-initial-guess PCG solve for manufactured
transpose problems.  It uses the explicit transpose action, maintains its
own diagnostics, and neither reads nor writes the production solver's
pressure warm start.  The solve executes under `torch.no_grad()`, so a PCG
iteration graph is not retained.

This is still an operator-level validation facility, not a functional
autograd implementation.  P9.7.2 does not add a custom backward rule, expose
an executable Channel functional runtime, qualify a pressure gradient, or
make activity differentiable.  Those claims remain reserved for P9.7.3 and
later slices.

## Qualification and boundaries

CPU equation tests cover:

- protocol identity and explicit non-delegating transpose assembly;
- complex linearity and zero-gauge invariance of primal and transpose;
- per-axis gradient/divergence adjoint identities;
- the full complex dot-product identity;
- measured Hermitian agreement for the current discretization;
- manufactured primal and transpose pressure recovery;
- zero right-hand side behavior, no iteration graph, and production warm-start
  isolation;
- preservation of the historical Phase 7 canonical solver SHA-256.

No H100 run is needed because this slice introduces no new production hot
path or functional timestep.  P9.7.3 is the next separately authorized
planning boundary: implement a custom implicit pressure adjoint and compare it
against a small-grid, fixed-iteration unrolled CPU oracle.  P9.7.2 does not
authorize P9.7.3 implementation, P9.7.4--P9.7.6, an H100 job, default
promotion, or Phase 9 closure.
