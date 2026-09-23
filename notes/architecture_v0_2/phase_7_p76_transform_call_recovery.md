# Phase 7 P7.6 Channel transform-call recovery

## Status

`READY_FOR_H100_RECOVERY_EXECUTION`

The first P7.6 H100 job, Slurm job `10839461`, produced all twelve requested
profiles and passed CUDA, runtime-identity, finite-state, paired-state, PCG,
memory, and performance checks available before the fail-closed stop. It
stopped because every profile reported 553 forward transforms over 50 measured
steps while the frozen contract expected 550.

## Root cause

The difference is exactly the production spectral-refresh schedule, not a GPU
or runtime discrepancy. The Channel integrator retains its production refresh
interval of 20 steps. Profiling begins after 10 warmup steps and ends at step
60, so scheduled refreshes occur at completed steps 20, 40, and 60. The five Q
components form one dynamic transform group, and every refresh adds one forward
transform call:

```
base forward calls       = 11 * 50 = 550
scheduled refresh calls  =  1 *  3 =   3
raw forward calls        =             553
raw forward calls/step   =           11.06
inverse calls            = 36 * 50 = 1800
```

The same `553/1800` result occurred for both runtime paths, both grids, and all
three trials. A local CPU reproduction using the same 10+50 step window also
gave 11.06 forward and 36 inverse transforms per step for both runtimes.

## Recovery contract

The profiler now records the raw forward/inverse totals, refresh-clock phase,
refresh counts before and after the measured window, dynamic transform-group
count, scheduled-refresh forward calls, and base forward calls after subtracting
only the clock-derived refresh contribution.

The fail-closed analyzer requires all of the following:

- refresh interval exactly 20;
- refresh phase 10 before and 0 after the window;
- exactly three expected and observed refreshes;
- exactly one dynamic transform group and three refresh forward calls;
- exactly 553 raw forward and 1800 raw inverse calls;
- exactly 11 base forward and 36 inverse calls per step;
- paired legacy/compiled raw call counts remain identical.

No solver, integrator, runtime, model, boundary condition, production default,
or scientific path is changed by this recovery. It corrects the profiler schema
and frozen qualification contract so that scheduled production work is
classified explicitly instead of being treated as an unexplained regression.

## Next qualification

Run a new single H100 recovery job from the committed recovery tooling. Repeat
the twelve profiles under the corrected schema, then—only after all profile
gates pass—continue the already frozen 100-step trajectory, same-runtime
restart, negative checkpoint, and 1000-step architecture-equivalence gates.
Do not promote `compiled_channel_v2`; `legacy_channel` remains the default.
