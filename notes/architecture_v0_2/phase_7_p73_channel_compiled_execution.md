# Phase 7 P7.3: opt-in compiled Channel execution

Status: `PASS_P7_3_CHANNEL_COMPILED_EXECUTION`.

P7.3 adds the first state/workspace/StepProgram execution path for the
rectangular Channel without connecting it to production.  The tensor-free
`pssolver.planning.channel_compiled_v2` declaration freezes the evolved and
algebraic component layout, each component group's native modal parity, the
persistent pressure state, semantic workspace requirements, and the exact
semi-implicit Euler operation order.  It also records the operations that are
forbidden from the timestep hot loop.

The execution adapter is deliberately available only by direct import from
`pssolver.experimental.channel_compiled_v2`.  It consumes the P7.1
`ChannelActiveNematicRunSpec`, constructs the P7.2 canonical Channel
Stokes/Schur solver, and replaces only the integrator binding with a
state-backed semi-implicit Euler implementation.  The physical and spectral
Q tensors are zero-copy views of the existing field storage.  The pressure
warm-start tensor is explicit persistent algebraic state and remains
identity-bound to the canonical Stokes solver.

The pre-bound StepProgram preserves the legacy Channel order:

1. prepare the algebraic Stokes state;
2. invoke the pre-update callback;
3. evaluate the explicit Q right-hand side;
4. add the timestep-scaled right-hand side in spectral space;
5. divide by the implicit denominator;
6. execute the pre-bound no-op dynamic-spectrum projection required by the
   frozen `dealias=none` Channel oracle;
7. inverse-transform the evolved Q groups;
8. perform any scheduled spectral refresh; and
9. commit progress.

The RuntimeWorkspace is explicit, stable across timesteps, and has a bounded
zero-slot plan at this stage.  This accurately describes P7.3 ownership: the
adapter introduces no new reusable tensor allocation, while the existing
constitutive, transform, and Stokes implementations still own their legacy
internal scratch.  The tensor-free declaration separately records those
semantic workspace categories so later migration can move them without
changing their ownership contract.

CPU qualification establishes:

- the FFT/DCT/DCT Q and pressure parity and FFT/DST/DST velocity parity;
- immutable, complete field and workspace declarations;
- a static, pre-bound operation order with no registry, capability,
  configuration, or fallback lookup in a timestep;
- zero-copy physical and spectral runtime state;
- explicit pressure warm-start capture and fail-closed restore;
- five-step byte-for-byte equality of physical fields, spectral fields, and
  pressure warm-start state against the legacy Channel oracle; and
- deterministic, JSON-serializable runtime metadata.

P7.3 intentionally preserves the current legacy Channel float32 construction
contract.  Float64 Channel qualification, production selection, checkpoint
and restart wiring, output ownership, and H100 performance evidence remain
future gates.  `ChannelRuntimePath` still contains only `legacy_channel`, the
experimental adapter is not exported from package roots, and `Channel.py` and
`pssolver.channel` retain their P7.2 source identities.

Local validation passed 81 targeted and compatibility tests.  The complete
suite passed 2,006 tests plus 8 subtests with no failure.

P7.3 makes P7.4 planning eligible.  It does not authorize P7.4
implementation, connect a compiled runtime to production, change a default,
authorize an H100 run, add a new boundary law, or modify Plane evidence.

The machine-readable result is
[phase_7_p73_channel_compiled_execution.json](phase_7_p73_channel_compiled_execution.json).
