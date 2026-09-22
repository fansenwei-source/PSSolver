# Phase 4 P4.3 convergence and restart qualification

Status: `P4_3_LOCAL_COMPLETE_REFERENCE_ONLY`.

P4.3 qualifies the disconnected scalar reference path with the temporal ladder
frozen in P4.0.  The unchanged spatial grid and final time use
`dt = (0.04, 0.02, 0.01, 0.005)`.  Final-time discrete L2 and Linf errors are
measured against the analytic cosine-mode solution.  Every error is finite and
strictly decreases.  The minimum observed pairwise L2 order exceeds `0.9` for
projected semi-implicit Euler and `1.8` for SBDF2.

This slice also introduces `provisional_generic_multistep_v1`, a
directory-based reference checkpoint that is separate from Plane checkpoint
v1.  Its manifest binds the exact integrator, fixed `dt`, model/discretization,
completed step, startup/multistep phase, representation validity, refresh
clock, current tensors, and complete SBDF2 history.  Every tensor record stores
shape, dtype, finite status, size, and SHA-256.  A `COMPLETE` marker binds the
canonical manifest and is written last; an existing target is never
overwritten.

A 12-step continuous run is byte-identical in physical state, native spectrum,
progress, refresh counters, and both history tensors to a 5+7 split/restart
run.  Negative tests reject a missing `COMPLETE`, missing history, tampered or
non-finite history, changed `dt`, a different integrator, an unsupported schema
version, an incompatible model/discretization, and a stale history clock before
any advance.

The convergence and checkpoint helpers remain direct-import-only CPU reference
tools.  They are not imported by a runtime, exported by a package facade, or
connected to Plane.  P4.4 may now introduce the separately qualified
two-component modal block operator.

The machine-readable record is
[phase_4_p43_convergence_restart.json](phase_4_p43_convergence_restart.json).
