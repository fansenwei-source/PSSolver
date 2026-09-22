# Phase 4 P4.6 memory-measurement recovery

Status: `P4_6_MEMORY_MEASUREMENT_RECOVERY_LOCAL_COMPLETE`.

The first registered H100 qualification (Slurm job `10837184`) passed every
CPU, package, CUDA, numerical, timing, workspace, and temporal-convergence
gate, but the formal validator rejected the peak-memory ratios.  At 131072
points the rebound/continuous peak-allocated and peak-reserved ratios were
`1.1055423426210595` and `1.1923076923076923`; at 1048576 points they were
`1.1099654771742460` and `1.1485148514851484`.  The frozen limit was `1.10`.

Inspection showed that the profiler retained the first role's complete GPU
final state in `final_states` while measuring the second role.  The added live
memory scaled linearly with point count and was therefore a profiler live-set
contamination, not workspace growth in the rebound stepper.  Peak-reserved
memory was additionally affected by unused caching-allocator blocks from the
previous role.

The recovery does not relax the `1.10` limit.  Each completed role is copied
to a CPU-only comparison snapshot and its GPU state, input state, and stepper
are released before the next role.  Unused allocator cache is cleared before
each timed interval, outside the timing region.  The profile schema is raised
to version 2, and the validator now requires explicit evidence that role live
sets were isolated, comparison snapshots were held on CPU, unused cache was
cleared, and each peak covers only the current role.

A local CUDA run at 131072 points gave a rebound/continuous median timing
ratio of `0.9963516772588991`, peak-allocated values of `42739200` and
`44826624` bytes, equal peak-reserved values of `67108864` bytes, and minimum
pairwise temporal order `2.0038328225707627`.  The code remains an opt-in
qualification canary and is still disconnected from Plane, runtime defaults,
and Phase 5.

The machine-readable recovery record is
[phase_4_p46_memory_measurement_recovery.json](phase_4_p46_memory_measurement_recovery.json).
