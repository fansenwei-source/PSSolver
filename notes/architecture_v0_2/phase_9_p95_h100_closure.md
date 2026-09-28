# Phase 9 P9.5: H100 batch-one functional closure

## Decision

P9.5 is complete with classification
`PASS_P9_5_H100_BATCH_ONE_FORWARD_VJP_MEMORY`.  Commit
`bdf9268a8b668eb79510dc0c51bae587142d26a8` passed the frozen CPU and single-H100
qualification without fallback, graph breaks, non-finite values, or a second
Slurm submission.  The authoritative machine record is
[phase_9_p95_h100_closure.json](phase_9_p95_h100_closure.json).

This result qualifies only batch size one for the periodic-box,
complete-stress Beris--Edwards activity-control functional runtime.  It does
not qualify larger batches, Channel pressure differentiation, an optimizer,
or a control-science result.

## H100 evidence

The one formal job, `10845880`, completed on an NVIDIA H100 PCIe with exit code
`0:0`.  All six CUDA-only tests passed and all 18 frozen profiles were
complete.  At R128 the production, functional-forward, and VJP mean timestep
costs were respectively 7.423167, 7.455183, and 30.245436 ms.  At R320 they
were 113.732030, 117.391653, and 431.254724 ms.  Functional/production mean
ratios were 1.004313 and 1.032178; VJP/functional ratios were 4.056967 and
3.673640.  Every performance and memory gate passed.

The stable validator-v2 directional errors were below `7e-8` on both grids.
All six named state/control dynamics and observation gradient paths were
finite, nonzero, and passed.  Production and functional R12 comparisons and
the frozen replay checks were bitwise.  The formal archive contains 43 files,
all verified by its checksum manifest.

## Capability consequence

P9.3 conservatively declared CUDA deterministic replay as `not_qualified`.
P9.5 has now supplied the missing H100 evidence, so the qualified runtime
declares `deterministic_replay="bitwise"` for both CPU and CUDA under one fixed
device, dtype, and execution identity.  This is a capability correction, not
a numerical-path change.

## Authorization

P9.6 is authorized to connect one independent real periodic consumer through
the provisional public functional API.  PSSolver remains independent of that
consumer.  The consumer owns its adapter, control parameterization, objective,
checkpoint schedule, and adjoint orchestration.  P9.7, P9.8, batch sizes above
one, and any production-default change remain unauthorized.
