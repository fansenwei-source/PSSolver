# Phase 7 P7.6: frozen Channel H100 closure plan

Status: `READY_FOR_H100_EXECUTION`.

P7.6 compares `legacy_channel` and explicit `compiled_channel_v2` from the
same clean commit.  It does not change the default, promote the compiled path,
alter the Channel equations or boundary spaces, reinterpret historical
snapshots, or modify Plane evidence.

## Frozen performance matrix

The two grids are the resolution-scaled small Channel `(128,20,20)` with
lengths `(32,5,5)` and the production oracle `(512,40,40)` with lengths
`(128,10,10)`.  Both use float32, `dt=0.01`, activity 5, seed 24, TF32 off,
10 warmup steps, and 50 measured steps.  Each grid has three balanced paired
trials: legacy/compiled, compiled/legacy, legacy/compiled.

Every pair must have identical initial and final state SHA-256, identical PCG
iteration history, identical transform counts, finite state, requested equal
to effective runtime, and no fallback.  The qualified local call contract is
11 forward and 36 inverse transforms per timestep.  Candidate/reference mean
and median timestep ratios must be at most 1.02, every individual ratio at
most 1.05, allocated and reserved memory ratios at most 1.05, and the PCG
iteration ratio at most 1.05.  Directional paired-win count is informational.

## Trajectory and restart gates

Both grids run matched 100-step continuous trajectories from the same
generated initial condition and require byte-identical final Q/u/p.  The
production grid additionally runs 50+50 same-runtime checkpoint continuations
for both paths; each resumed final Q/u/p must be byte-identical to its own
continuous result.  Cross-runtime restore and identity/checksum corruption
must still be rejected before target mutation.

After those gates pass, both runtimes run the historical production-sized
1,000-step Channel workload, saving every 100 steps.  All paired Q/u/p frames
must be byte-identical, diagnostics finite, pressure mean consistent with the
zero-mean gauge, COMPLETE valid, and worktree clean.  This is architecture
equivalence evidence, not a new physical benchmark or steady-state claim.

Only one H100 Slurm submission is authorized.  A pre-script bootstrap failure
does not become a scientific result, but no retry is automatic.  Any CPU,
CUDA, identity, finite-state, PCG, transform, memory, performance, trajectory,
restart, provenance, or checksum failure stops the job and forbids default
promotion.  Successful P7.6 still leaves `legacy_channel` as the default and
only makes Phase 8 planning eligible.

The machine-readable contract is
[phase_7_p76_h100_qualification_plan.json](phase_7_p76_h100_qualification_plan.json).
