# Phase 4 P4.6 final closure

Status: `PASS_PHASE4_CLOSURE_WITH_MEMORY_MEASUREMENT_RECOVERY`.

The schema-v2 H100 recovery completed the last Phase 4 gate.  The original
schema-v1 execution is retained as a valid failure record: its scientific,
timing, workspace, and convergence checks passed, while its memory check was
contaminated by a retained prior-role GPU state and allocator cache.  The
recovery isolated each role's live set without changing the registered `1.10`
time or memory limits.

On the H100, the rebound/continuous median timestep ratios were
`1.003408602806614` at 131072 points and `1.0019911480494244` at 1048576
points.  Peak-allocated ratios were `1.0488409703504042` and `1.0`; peak-
reserved ratios were `1.0` and `1.0327868852459017`.  Every physical and
native-spectral result was byte-identical, each workspace retained exactly
seven stable tensors, and no fallback, OOM, NaN, Inf, or CUDA error occurred.
The minimum observed temporal order was `2.0038328225700184`.

Phase 4 is therefore complete.  The architecture now has a qualified
constant-step SBDF2 contract, persistent multistep history, a bounded
two-component modal block operator, continuous/checkpoint restart identity,
and an H100-qualified combined reference canary.  All of these remain opt-in
reference capabilities: Plane, the production runtime and checkpoint format,
public APIs, and production defaults are unchanged.

This record makes Phase 5 planning eligible.  It does not authorize Phase 5
implementation, connect the canary to Plane, promote `separated_canary`, or
change a production default.

The machine-readable record is
[phase_4_p46_final_closure.json](phase_4_p46_final_closure.json).
