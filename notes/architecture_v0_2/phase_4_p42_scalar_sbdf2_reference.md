# Phase 4 P4.2 scalar SBDF2 reference path

Status: `P4_2_LOCAL_COMPLETE_REFERENCE_ONLY`.

P4.2 adds a direct-import-only, CPU float64 reference workflow in
`pssolver.integrators.scalar_reference`.  It solves a one-dimensional periodic
scalar reaction--diffusion problem whose diffusion term is implicit and whose
linear reaction term is explicit.  A single cosine mode supplies an analytic
solution without introducing a production model, geometry, or checkpoint
dependency.

The implementation supports the declared projected semi-implicit Euler method
and constant-step SBDF2.  An SBDF2 run starts with exactly one Euler step.  The
successful startup stores the original spectrum and its explicit RHS as the
first complete history level; subsequent steps use the frozen SBDF2 formula.
History validation rejects missing, stale, layout-incompatible, or changed-dt
state before arithmetic begins.

Each step is a functional state transition.  All candidate spectra, physical
fields, refresh counters, progress, and replacement history are constructed
without modifying the input state.  Progress is prepared only after the
optional refresh and history is prepared last.  A pre-update callback runs
after algebraic preparation and before current-RHS evaluation.  A diagnostic
stage observer permits an exception to be injected after every named stage;
tests verify that every such failure preserves the complete input state and
history.

The reference path is deliberately strict: CPU only, physical `float64`,
native spectral `complex128`, fixed `dt`, one even periodic grid, and no
implicit fallback.  It remains disconnected from `RuntimeState`,
`StepProgram`, Plane, package-root exports, and production checkpoint v1.
P4.3 will use this frozen path for temporal-order and split/restart
qualification; those claims are not made in P4.2.

The machine-readable record is
[phase_4_p42_scalar_sbdf2_reference.json](phase_4_p42_scalar_sbdf2_reference.json).
