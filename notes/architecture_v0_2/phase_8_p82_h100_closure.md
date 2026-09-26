# Phase 8 P8.2: periodic complete-stress H100 closure

Status: `PASS_P8_2_PERIODIC_COMPLETE_STRESS_H100_QUALIFICATION`.

P8.2 is complete and is eligible for P8.3 planning. The original local
candidate record remains unchanged: it records what was known before H100
qualification, while this document records the independent HPCC closure.

The final qualification used commit
`44a76b36a86e221fde3cfe0ef59d5a9a17a091c7` from a detached, clean worktree.
The single authorized H100 job, `10842576`, completed with exit code `0:0` on
an NVIDIA H100 PCIe. The installed package was imported outside the source
checkout, the requested and effective runtime were both `periodic_spectral`,
runtime fallback was false, TF32 was disabled, all three CUDA-only tests
passed, and the manufactured periodic-Stokes gate passed.

## Evidence chain and attribution

The closure preserves the complete fail-closed history rather than rewriting
earlier stops as successes:

1. The original attempt stopped in an external helper because it accessed
   `compiled.run_spec` instead of `compiled.application_request`.
2. Recovery v1 stopped in an external helper because it accessed
   `model.nonlinear_model` instead of `model.nlmodel`.
3. Recovery v2 ran while a foreign VLLM process occupied the GPU. Its first
   R128 timing window had 9.3% coefficient of variation, and its transform
   counts also included post-measurement observation work.
4. The analysis-only adjudication identified those two measurement-contract
   errors without modifying solver code or scientific output.
5. Recovery v3 used an exclusive, uncontaminated H100 allocation, separated
   timed-loop counters from observation counters, and passed every frozen
   scientific, restart, negative, provenance, and stability gate.

The four preceding archives and the final archive were checksum-verified and
remained immutable. The final archive contains 265 verified entries and a
`COMPLETE` marker whose content is the final classification.

## Uncontaminated performance evidence

Each grid used 30 warmup steps, 100 measured steps, and three independent
trials. Nineteen GPU-process audits found no foreign compute process before,
during, or after any of the six trials.

For R128, the aggregate mean timestep was 5.168264 ms, the aggregate median
was 5.146752 ms, and mean throughput was 193.493057 steps/s. Trial CV values
were 1.557462%, 2.472408%, and 1.687311%. The maximum/minimum ratios were
1.011821050 for trial means and 1.011983544 for trial medians.

For R320, the aggregate mean timestep was 86.907037 ms, the aggregate median
was 86.873615 ms, and mean throughput was 11.506549 steps/s. Trial CV values
were 0.108821%, 0.061791%, and 0.051802%. The maximum/minimum ratios were
1.000770569 for trial means and 1.000727134 for trial medians.

Peak allocated/reserved memory was 394224640/650117120 bytes for every R128
trial and 6132706816/9619636224 bytes for every R320 trial. All samples were
finite and positive; graph breaks, compile fallbacks, runtime fallbacks, OOMs,
NaNs, infinities, and CUDA errors were zero. There is no prior periodic
runtime performance baseline, so these measurements establish a qualified
reference and do not support a speedup or benchmark claim.

## Transform-call contract

For all six profiles, the formal 100-step measured-loop delta was exactly 700
forward and 3100 inverse calls, or 7.0/31.0 calls per step. The first measured
step used 7/31 calls and the remaining 99 steps used 693/3069 calls. The
post-measurement observation used another 6/21 calls; those calls were
recorded separately and did not contaminate the timed-loop contract. R128 and
R320 had identical transform topology and raw deltas.

## Production, restart, and rejection gates

Four production trajectories completed: R128 continuous 100-step, R128 first
50-step plus checkpoint, R128 resumed to step 100, and R320 continuous
100-step. Every run had valid `COMPLETE`, runtime identity, float64 Q/u/p
shapes, finite arrays, pressure mean at most `1e-12`, mean-velocity norm at
most `1e-12`, and maximum divergence at most `1e-10`.

The R128 continuous and resumed Q, velocity, and pressure files were
byte-for-byte identical. Six checkpoint corruption classes—tensor bytes,
checksum metadata, shape/dtype record, runtime path, runtime identity, and
backend identity—were rejected before a timestep, without mutating the target
state or writing `COMPLETE` in a negative copy.

## Scope after closure

This qualification closes exactly the registered combination
`complete_stress_beris_edwards + periodic_box` with the
`periodic_spectral` runtime. It does not promote a compiled runtime, change the
Plane or Channel production defaults, establish a paper benchmark, authorize
Phase 9, or silently expand the registered combination matrix. P8.3 planning
is eligible; P8.3 itself has not started.

The machine-readable authority is
[phase_8_p82_h100_closure.json](phase_8_p82_h100_closure.json).
