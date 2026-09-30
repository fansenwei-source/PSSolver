# Phase 9 P9.7.1: Channel functional declarations

## Result

P9.7.1 adds a non-executable, provisional functional declaration for the
complete-stress Beris--Edwards rectangular Channel.  The machine-readable
authority is
[phase_9_p971_channel_functional_declarations.json](phase_9_p971_channel_functional_declarations.json).
This slice changes PSSolver's interface declarations only.  It does not
modify the independent Control project, the frozen `pssolver.control` oracle,
the production Channel runtime, the pressure solver, or any default.

The declaration is public only through `pssolver.functional`.  It is not
re-exported from the stable package root while the API remains
`0.1-provisional`.

## Generic declaration boundary

`FunctionalRuntimeDeclaration` binds five facts before a runtime exists:

- one canonical `FunctionalRuntimeConstructionRequest`;
- the exact ordered `FunctionalStateSpec`;
- a fail-closed `FunctionalCapabilitySet`;
- a JSON-safe `FunctionalRuntimeIdentity` and canonical SHA-256;
- whether execution is actually available.

A non-executable declaration is forbidden from claiming a pure step, combined
observation, deterministic replay, differentiability, checkpoint bridge, JVP,
VJP, or differentiable input.  This prevents an interface declaration from
being mistaken for a qualified runtime.

## Channel request, state, control, and observations

`channel_activity_functional_declaration()` accepts only the registered
complete-stress rectangular-Channel application, batch one, float64,
full-complex spectral storage, TF32 off, and eager pointwise kernels.  It runs
the tensor-free public compiler but does not read the initial snapshot,
construct a solver, allocate runtime state, execute PCG, or create an output
directory.

The future state is the ordered tuple `(q_physical, q_spectral)`.  The
physical tensor uses compact five-component Q layout.  The spectral tensor
uses the native periodic-x/Neumann-y/Neumann-z full-complex basis.  Pressure
guess, Q-gradient cache, diagnostics, and progress counters are not state.

The only control is the non-negative, cell-centered activity field.  Its
identity states that `beta * activity * Q` is formed before the projected
component-basis divergence.  There is no implicit dtype/device conversion or
clipping.  The declared observations are input-state `Q`, velocity, and
pressure.  Only `Q` is available terminally without a control.

## Pressure identity without a gradient claim

The identity records the matrix-free no-slip Channel pressure PCG, its
tolerance and iteration controls, the zero-mean gauge, and all Q/velocity/
pressure boundary spaces.  It explicitly distinguishes the unchanged
production warm start from the future functional zero initial guess:

- production warm start: enabled, unchanged;
- functional warm start: disabled;
- functional pressure initial guess: zero;
- transpose action: not implemented;
- pressure gradient: not qualified.

Consequently P9.7.1 cannot be passed to an executable Channel functional
factory.  The existing periodic builder continues to reject it before runtime
allocation.

## Ownership and next slice

Only the PSSolver repository changes in this slice.  No adapter, objective,
checkpoint scheduler, adjoint orchestration, optimizer, or experiment is
added to PSSolver, and no PSSolver import of the independent Control project
exists.

P9.7.1 completes the declaration layer only.  P9.7.2 is the next separately
authorized slice: define an explicit Channel pressure-transpose protocol and
qualify operator linearity, zero-mean invariance, dot-product identity, and
manufactured primal/transpose solves.  P9.7.1 does not authorize that
implementation, an H100 job, P9.7.3--P9.7.6, or Phase 9 closure.
