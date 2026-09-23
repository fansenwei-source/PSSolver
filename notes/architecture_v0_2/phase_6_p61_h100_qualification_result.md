# Phase 6 P6.1 H100 qualification result

Classification: `PASS_PHASE6_H100_CANDIDATE_PERFORMANCE_EQUIVALENT`.

Qualification commit:
`9dcb9dc0a00b7fa1b12b200e83aaf0c6f21eb4db`.

The immutable profiler source is commit
`bad9916ac1cae4969372712edf88d106d529d584`, Job `10837343`, and manifest
SHA-256
`ba00f7f5bc64cd5ae022d53d6eab3505636b0384fd047b8650e91b7d3167133c`.
That Job remains classified by its original frozen contract as
`FAIL_PHASE6_H100_CANDIDATE_GATE`; this record does not rewrite it.

The authorized recovery Job `10838294` completed on one NVIDIA H100 PCIe with
ExitCode `0:0`.  The old manifest reverified `1670/1670`, the recovery
manifest verified `1241/1241`, and the recovery manifest's own SHA-256 is
`488a6ece0f2876c5d2ae2db841c0f7bb1b59251345edf7c05082f84d6c39db44`.

The complete CPU gate passed `1938` tests and 8 subtests, with the one
pre-existing optional `nematics3d` skip and exactly three authorized
CUDA-only deselections.  Those three tests then passed on the H100.

The twelve original profiler records classify both R128 and R320 as
`performance_equivalent`: all mean, median, individual paired-trial, peak
allocated-memory, and peak reserved-memory ratios remain within the frozen
two-percent non-regression envelope.  No new profiler was run.

Eight fresh continuous/restart trajectories completed.  Both cross-runtime
100-step comparisons and all four same-runtime continuous-versus-restart
comparisons were byte-identical.  The four cross-runtime/tampered restart
negative gates rejected before mutation, and all 13 auxiliary gates passed.

Consequently P6.1 is complete and the separately frozen P6.2 T=400 long-run
architecture-equivalence stage is eligible.  This result does not promote
`compiled_v2`, change the implicit default, establish physical stationarity,
or establish strict equivalence to the inertial Shendruk model.
