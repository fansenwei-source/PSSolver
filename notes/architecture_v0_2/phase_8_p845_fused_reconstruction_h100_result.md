# P8.4.5 fused-reconstruction H100 result and diagnostic continuation

## Frozen result

Job `10843818` evaluated candidate
`0e883b3efedf65c5b9cfa01cb1cfbe7025c976bc` against parent
`046ad5b12b31ea627f8f87fe4286ba6dae22393e`.  The frozen outcome remains
`FAIL_P8_4_5_PLANE_STATIC_LIFTING_H100_CLOSURE`; no threshold, production
default, runtime selection, or phase authorization is changed by this note.

The recovery did remove every persistent full-domain lift, affine-Laplacian,
linear-correction, and physical-Q workspace.  All twelve H100 profiles were
finite and retained compiled pointwise execution, zero graph breaks, no
fallback, and `7/32` forward/inverse transforms per step.  Wall residual,
manufactured convergence, timestep, live allocated memory, and exact
same-runtime checkpoint/restart gates passed.

At R320 the lifting case was faster (`0.989166x` mean timestep), used
`1.052999x` peak allocated memory, and used `0.999091x` peak reserved memory.
At R128 the timestep (`1.014430x`) and peak allocated memory (`1.049557x`)
passed, but peak reserved memory was `1.297297x`, above the unchanged `1.10`
limit.  The R128 reserved increase was `138,412,032` bytes while live peak
allocation increased by only `21,106,688` bytes; the larger R320 case did not
show this reserved-memory increase.  This pattern is consistent with a CUDA
allocator or compiled-kernel cache size-class effect, but that attribution is
not yet accepted as a qualification result.

The same H100/PyTorch 2.5.1 environment also produced different parent and
candidate final SHA-256 values after three compiled steps, despite identical
initial Q, transform counts, lifting restart metadata, and exact
continuous/resumed identity within the candidate.  The fused pointwise graph
can change floating-point operation boundaries and last-bit rounding.  A hash
mismatch alone does not quantify whether the difference is roundoff-sized or
scientifically material, so the cross-implementation identity gate remains
failed pending direct array comparison.

## Analysis-only continuation

`benchmarks/diagnose_plane_static_lifting_recovery.py` adds three read-only
diagnostic commands:

1. `capture-state` runs a fixed short trajectory and saves every internal and
   reconstructed physical field to an NPZ archive together with exact runtime,
   package, Git, initial-state, and output-file identities.
2. `compare-states` compares two archives field by field and reports mismatch
   count, relative L2, RMS, Linf, and maximum float64 ULP distance.  It applies
   no tolerance and makes no qualification decision.
3. `diagnose-memory` records allocated, requested, active, reserved, and
   inactive-split CUDA allocator counters after construction, cleanup, warmup,
   the measured profile, and physical observation in a fresh process.

The next H100 task is analysis-only.  It must use isolated parent and candidate
packages, capture the same three-step strong-planar trajectory, compare the
arrays, and run fresh-process R128 control/lifting allocator diagnostics.  It
must not rerun the complete qualification, relax either memory limit, change
the cross-version contract, modify source, or enter P8.5.

## Decision rule after diagnostics

- If differences are roundoff-sized and longer matched trajectories remain
  within a separately preregistered scientific tolerance, preserve exact
  byte identity for same-runtime restart but use an explicit numerical
  equivalence contract for cross-implementation comparison.
- If differences are larger than roundoff or grow unexpectedly, restore an
  explicit physical-Q rounding/materialization boundary before qualification.
- If R128 excess reserved bytes are inactive/cached while active and allocated
  bytes remain within the existing limit, revise the future metric only through
  a documented contract change; do not retroactively convert Job `10843818`
  into a pass.
- If active memory itself accounts for the excess, continue implementation
  optimization before another closure attempt.

Nonhomogeneous Neumann data, P8.5, Phase 9, compiled-runtime promotion, and
production-default changes remain outside scope.
