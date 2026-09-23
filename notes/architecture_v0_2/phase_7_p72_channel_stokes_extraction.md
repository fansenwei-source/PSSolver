# Phase 7 P7.2: Channel Stokes/Schur extraction

Status: `PASS_P7_2_CHANNEL_STOKES_SCHUR_EXTRACTION`.

P7.2 moves the existing rectangular-Channel modal Stokes/Brinkman solve into
the model-independent
`pssolver.linear_solvers.stokes.channel_no_slip` implementation boundary.  The
canonical `ChannelNoSlipModalStokesSolver` owns the mixed FFT/DST/DCT basis
changes, diagonal velocity Helmholtz inverse, pressure Schur operator,
zero-mean pressure projection, preconditioned conjugate-gradient solve,
pressure warm start, and pressure diagnostics.  It does not import an active-
nematic model or construct active force.

The numerical contract is unchanged.  All velocity and force components use
periodic/Dirichlet/Dirichlet space, pressure uses
periodic/Neumann/Neumann space, and the pressure coefficient at modal index
(0,0,0) is removed as the gauge.  Both bounded velocity axes therefore retain
DST parity while both bounded pressure axes retain DCT parity.  The copied
implementation deliberately preserves the legacy allocation and operation
order.

`pssolver.channel.ModalSaddleStokesCompute` remains at its historical import
path and keeps its constructor and active-force `forward` behavior.  It is now
a thin compatibility subclass of the canonical linear solver.  Consequently
`build_active_nematic_channel` and the unchanged `Channel.py` entry point keep
their existing public behavior.  The Stage G experimental geometry adapter
now instantiates the canonical linear solver directly instead of manufacturing
a legacy solver-shaped namespace.

Equation-level float64 CPU tests independently establish:

- the FFT/DST/DST velocity and FFT/DCT/DCT pressure basis contract;
- exact registered-operator identity between the compatibility facade and the
  canonical solver;
- byte-identical force-solve results from fresh legacy and canonical objects;
- zero pressure gauge, incompressibility, and the modal wall-momentum balance;
- recovery of a manufactured pressure through the Schur operator;
- exact zero-force behavior and pressure diagnostics;
- fixed-iteration pressure work-count behavior; and
- fail-closed coefficient and dimensionality validation.

Existing Stage G trajectory, residual, operation-order, pressure warm-start,
restart, and geometry-policy tests continue to pass.  The P7.0 hashes remain a
historical identity of the pre-extraction oracle; the P7.2 record separately
binds the new canonical implementation and compatibility facade.  `Channel.py`
is still byte-identical to the P7.0 entry-point oracle.

Local validation passed 61 targeted and compatibility tests.  The complete
suite passed 1,993 tests plus 8 subtests with no failure.

P7.2 makes P7.3 planning eligible.  It does not create a compiled Channel
runtime, connect the P7.1 run specification to production, change a default,
authorize an H100 run, add a new boundary law, or modify Plane evidence.

The machine-readable result is
[phase_7_p72_channel_stokes_extraction.json](phase_7_p72_channel_stokes_extraction.json).
