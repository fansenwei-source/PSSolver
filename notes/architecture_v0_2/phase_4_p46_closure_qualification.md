# Phase 4 P4.6 closure qualification

Status: `P4_6_LOCAL_COMPLETE_H100_CLOSURE_PENDING`.

P4.6 closes Phase 4 only after the P4.5 opt-in SBDF2/modal-block canary passes
source, package, CPU, checkpoint/restart, negative-gate, and H100 execution
contracts.  This local slice adds the frozen H100 profiler and a fail-closed
analysis validator; it does not itself claim H100 qualification.

The performance comparison is contract-equivalent.  Role A advances one
complete-history SBDF2 state continuously.  Role B constructs an equivalent
stepper, validates and rebinds the same complete state/history boundary, and
then advances the identical number of steps.  Both roles use float64,
`closed_form_2x2`, TF32 disabled, the same analytic model, and balanced trial
order.  This avoids comparing SBDF2 with Euler or using a different linear
algebra kernel as a performance baseline.

The registered H100 points are `131072` and `1048576` physical points.  Each
role uses five warmup steps, thirty measured steps, and three balanced trials.
The rebound/continuous median time ratio and maximum peak allocated/reserved
memory ratios may not exceed `1.10`.  Every trial must retain the seven fixed
workspace tensors, remain finite, use the closed-form implementation, and
produce byte-identical final physical and spectral states between roles.

The same job also executes the CUDA-only device-identity test and the fixed
four-level float64 temporal convergence ladder.  Every pairwise L2 order must
be at least `1.8`.  The evidence validator rejects duplicate JSON keys,
non-finite values, wrong commit/device/H100 identity, TF32, incomplete or
unbalanced trials, changed workspace count, unstable pointers, numerical
differences, recomputed-statistic mismatches, excessive time or memory, and
unauthorized Phase 5/default claims.

Before the single H100 job, the closure task must run the complete CPU suite
with the one CUDA-only node deselected, build a clean sdist and wheel, install
the wheel outside the source checkout, import both P4.5 modules from the wheel,
and complete a CPU run plus checkpoint/restart smoke.  The CUDA-only test must
then pass—not skip—inside the H100 job.

A passing H100 profiler and validator authorize a later evidence-only Phase 4
closure record.  They do not by themselves authorize Phase 5, connect the
canary to Plane, promote a public API, or change a production default.

The machine-readable record is
[phase_4_p46_closure_qualification.json](phase_4_p46_closure_qualification.json).

The first H100 execution exposed a live-set contamination in the memory
profiler.  Its diagnosis and non-relaxing recovery are recorded in
[phase_4_p46_memory_measurement_recovery.md](phase_4_p46_memory_measurement_recovery.md).
