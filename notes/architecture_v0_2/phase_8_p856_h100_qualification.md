# Phase 8 P8.5.6: finite-Q anchoring H100 closure

Status: `PASS_P8_5_6_FINITE_Q_ANCHORING_H100_CLOSURE`.

The qualified source is
`74a5226f6798cc311d3b839624b85349e3647fe8`; the core CUDA implementation is
`c8028845293247290a2e9aa51ce8847206768bfa`, and the P8.5.5 baseline is
`2c99942f37149aaa92791ab8616697707f6abfbb`.  The corrected qualification
contract is recorded by
`5f1ba3f2e24f0070f25e39931e0a7c62bc86990b`.

P8.5.6 closes the numerical, device, restart, performance, memory,
operation-count, conditioning, package, and provenance gates for the
five-component finite-Q anchoring relaxation pilot.  It does not connect
Robin Q to the complete Beris--Edwards/Stokes timestep, the public
`Simulation` compiler, or a production runtime selector, and it changes no
production default.

## Recovery and CPU gate

The first attempt stopped before any GPU submission because its CPU contract
omitted one existing CUDA-only static-lifting device test.  The corrected
contract deselected exactly six CUDA-only tests on the login node and moved
all six to the real-H100 gate.  No scientific result was discarded and the
original 17-file failure archive remained checksum-clean.

The corrected CPU suite passed `2391 passed, 1 skipped, 6 deselected,
8 subtests passed`.  The only skip was the existing optional `nematics3d`
dependency test.  The detached source worktree ended clean with no tracked,
untracked, ignored, Python-cache, or pytest-cache artifacts.

The qualification wheel was built from the frozen source and imported from a
non-editable isolated environment outside the checkout.  Its SHA-256 is
`17d9d698bd2400d6c7005a6518b9201dc856d2f52fadb8b08c14313080c40fbc`;
source shadowing was absent.

## H100 evidence

The single authorized job, `10844444`, completed on `gpu-h100-4-0` with
`ExitCode=0:0`, no requeue, and no restart.  It used one NVIDIA H100 PCIe,
PyTorch 2.5.1+cu121, CUDA 12.1, driver 550.54.14, and disabled both matmul and
cuDNN TF32.  All six transferred CUDA-only tests passed on the allocated
device without skip, deselection, xfail, or fallback.

All twelve profiles completed: two grids (`128x128x32` and `320x320x80`),
three paired trials, and both the per-component scalar reference loop and
aggregate five-component runtime.  Every paired trial had identical initial
and final physical-Q SHA-256 identities.  Every report was finite, clean,
installed-package sourced, fallback-free, graph-break-free, and showed no
root solve, basis construction, or matrix factorization in the timed loop.

The exact per-step operation contract held in all reports:

```text
forward transforms = 10
inverse transforms = 5
Helmholtz solves = 5
Helmholtz applies = 0
```

The largest sampled-basis condition number was
`1.2099038387330427`, far below the frozen `1e10` bound.

## Performance and memory

At `128x128x32`, the aggregate/reference mean and median timestep ratios were
`1.070141406632` and `1.070852792396`.  At `320x320x80`, they were
`1.069896692523` and `1.070174058065`.  These remain below the deliberately
conservative `1.15` qualification limit.  The aggregate path is therefore
about seven percent slower than the scalar-reference organization in this
pilot, but it does not constitute a performance regression under the frozen
contract and provides the required aggregate state/restart semantics.

Peak allocated, active, and reserved memory ratios were exactly `1.0` at
both grids.  The repository analyzer returned
`PASS_P8_5_6_PROFILE_MATRIX` with all twelve reports present.

## Qualification boundary

The final recovery archive contains `COMPLETE`, and its 34-entry checksum
manifest passed in full.  The archive root is
`/home/fansenwei/pssolver_phase8_p856_finite_q_h100_74a5226_20260927_recovery_v1`;
the manifest SHA-256 is
`1db095d21d6375b3192fa048e1b7b33a0bef960a925206e3e025d0847c4d59d0`.

P8.5 is now closed and P8.6 planning is eligible.  This result makes no paper
benchmark claim, does not authorize Phase 9, does not promote a compiled
runtime, and does not authorize a production default or public-runtime
connection.  P8.6 must separately close the Phase 8 capability catalog,
installed-package behavior, rejection matrix, metadata, checkpoint/restart,
numerical-oracle, and cumulative H100 evidence.
