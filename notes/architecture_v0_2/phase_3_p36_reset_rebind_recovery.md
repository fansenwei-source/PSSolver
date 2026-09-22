# Phase 3 P3.6 reset/rebind recovery

Status: `FAIL_P3_6_RESET_REBIND_HARNESS_IDENTITY_RECOVERY_PENDING`.

The first reset/rebind recovery reached the H100 legacy-production smoke and
called reset successfully, but the external qualification harness then failed
three identity/control assertions.  The failure does not establish a solver,
CUDA, numerical, scientific, or performance defect.

The harness retained only integer `id()` values and CUDA storage addresses
for the pre-reset objects.  Because it released the objects themselves,
Python and the CUDA allocator were free to reuse those numbers.  An address
or integer-ID inequality after deallocation is therefore not a valid object
identity contract.  The corrected test must retain strong references to the
old state, workspace, tensors, and storages while comparing them with the
rebound objects.

The harness also reused the same initial tensor objects for the evolving
runtime, the reset call, and the fresh-construction control.  A runtime may
legitimately evolve those adopted tensors in place, so this did not provide
an independent fresh initial state.  The corrected test must create separate,
non-overlapping clones from a pristine immutable source for every role and
verify the source hashes before and after execution.

The canonical Phase 3 reset tests also fix two details that the recovery must
not reinterpret.  Reset replaces `RuntimeState` and rebinds its physical and
spectral views to the solver's newly built fields, but the bounded workspace
is allocated once and remains the same workspace object.  Reset/rebind also
preserves the integrator progress counters.  A fresh-construction control must
therefore have its progress aligned through the qualified progress-restoration
surface before a post-reset evolution is compared; zero-progress fresh state
is not the reset progress oracle.

All previously qualified performance, continuous-trajectory, and restart
evidence remains valid.  The next recovery remains limited to the corrected
reset/rebind smoke and the eleven negative gates.  Phase 3 and Phase 4 remain
open and unauthorized, respectively.

The machine-readable record is
[phase_3_p36_reset_rebind_recovery.json](phase_3_p36_reset_rebind_recovery.json).
