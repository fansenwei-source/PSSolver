# Phase 8 P8.4.5: static Plane lifting H100 closure plan

Status: `READY_P8_4_5_H100_QUALIFICATION_NOT_EXECUTED`.

The P8.4.4 implementation at baseline commit
`70eb409c26ad335b9b6519501e4ecd5fa9f0d3e2` connects field-neutral static
lifting to the Plane legacy production runtime, exposes physical Q, and
checkpoints the homogeneous remainder.  P8.4.5 changes no lifting equation
and no production default.  It freezes the final, single-H100 non-regression
contract and supplies repository-owned measurement and analysis helpers.

## Scientific case

The qualified candidate uses strong planar Q anchoring with scalar order
`S=0.6`, director `(1,0,0)` at the lower wall and `(0,1,0)` at the upper
wall.  Unlike equal homeotropic Q at oppositely oriented walls, this case has
a nonzero affine wall-normal slope and therefore exercises reconstruction,
gradient correction, physical observation, and restart identity.

The performance control uses homogeneous Neumann Q with the same complete
stress Beris--Edwards model, Plane geometry, initial random field, dtype,
dealiasing, transform implementation, timestep, and legacy production
runtime.  It is a performance control, not a claim that the two boundary
value problems have the same physical trajectory.

## Frozen evidence matrix

One H100 job must run three paired trials at `128x128x32` and `320x320x80`.
Each trial invokes the repository helper once for the homogeneous control and
once for strong planar lifting.  The order must be balanced across trials.
The same job also runs:

1. the CUDA manufactured oracle at 16, 32, and 64 wall-normal points;
2. a `128x128x32`, 10+10-step lifted checkpoint/restart comparison;
3. the fail-closed repository analyzer.

The package must be installed into an isolated target or environment and all
imports used by the job must resolve to that installed package.  The source
checkout is evidence and command authority only; shadow imports fail.

## Frozen gates

- wall-value `L_inf <= 1e-12` under analytic half-cell extrapolation;
- manufactured minimum rate `>= 1.95`;
- 7 forward and 32 inverse transforms per profiled timestep;
- no runtime fallback, compile fallback, graph break, NaN, Inf, CUDA error,
  or OOM;
- lifting/control paired mean and median timestep aggregate ratios `<= 1.15`;
- maximum peak allocated and reserved memory ratios `<= 1.10`;
- continuous and restored Q/u/p, including reconstructed physical Q, are
  byte-for-byte identical;
- all reports, logs, plans, installed artifacts, and provenance are covered
  by a final SHA-256 manifest and a `COMPLETE` marker written only after every
  gate passes.

The 15% timestep ceiling is a feature-overhead bound, not a performance
promotion criterion.  The 10% CUDA-memory ceilings are intended to detect
duplicate full-domain lift storage or hot-loop allocation.  Static lift
construction occurs outside the measured window.

## Stop and authorization rules

Only one formal H100 submission is authorized.  Any identity, CPU, CUDA,
numerical, performance, memory, transform-count, restart, analyzer, or
checksum failure stops the task without retry.  A scheduler/bootstrap failure
before the job body begins must be reported and is not silently retried.

P8.4.5 does not authorize nonhomogeneous Neumann data, Robin or finite
anchoring, Channel lifting, compiled-runtime lifting, default changes,
P8.5 implementation, Phase 9, or a paper benchmark claim.

The machine-readable contract is
[`phase_8_p845_h100_qualification_plan.json`](phase_8_p845_h100_qualification_plan.json).
