# P8.4.5 static Plane lifting H100 qualification

Status: `PASS_P8_4_5_PLANE_STATIC_LIFTING_H100_CLOSURE`.

Job `10843953` qualified commit
`6a9e4e0cb9b1fdd4f0255ca71aeedfb168748bc2` on one NVIDIA H100 PCIe.  The
complete CPU, installed-package, CUDA, 12-profile, manufactured, restart,
cross-version numerical-equivalence, provenance, and checksum gates passed.
The final archive contains 2,739 verified entries and has manifest SHA-256
`11212eddf626a9adf07cba743cfd49429df5dac43adb787b655679cb2a8dafdf`.

## Numerical result

The lifted same-runtime 10+10-step restart is byte-for-byte identical for
internal Q/u/p and reconstructed physical Q.  The independent wall-normal
manufactured test has minimum convergence rate `1.9986095691686334`, and all
strong-wall residuals are zero at the measured precision.

The 20-step compiled comparison against reference commit
`046ad5b12b31ea627f8f87fe4286ba6dae22393e` used identical initial Q and
validated all 18 internal/physical fields separately.  Its largest fieldwise
relative L2 is `6.650683458073026e-15`, and its largest Linf is
`2.220446049250313e-16`, below the preregistered `1e-12` limits.  ULP and byte
mismatch counts remain diagnostics; they do not replace fieldwise norms.

## Performance and memory result

At R128, lifting/control mean and median timestep ratios are `1.0186896994`
and `1.0107321865`; peak allocated and active ratios are both
`1.0495574285`.  At R320, the corresponding ratios are `0.9888881647`,
`0.9893546372`, and `1.0529987104`.  All hard limits pass.

Reserved ratios (`1.2972972973` at R128 and `0.9990906335` at R320) are
recorded as allocator diagnostics under the corrected contract.  The runtime
owns no persistent full-domain physical-Q reconstruction workspace, retains
`7/32` transforms per step, and reports no fallback or graph break.

## Scope

P8.4.5 is complete.  This result qualifies field-neutral static Dirichlet
lifting and the Plane strong-planar Q specialization on the legacy production
runtime.  It does not add prescribed nonzero Neumann flux, finite/Robin
anchoring, Channel lifting, compiled-runtime promotion, a production-default
change, or a paper benchmark result.

P8.5 planning is now eligible.  P8.5 implementation, nonhomogeneous Neumann
work, and Phase 9 remain separately unauthorized.
