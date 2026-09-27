# P8.4.5 fused physical-Q reconstruction recovery

## Status

`READY_P8_4_5_FUSED_RECONSTRUCTION_MEMORY_RECOVERY`

Job 10843753 verified that compact lift profiles, compact affine Laplacians,
and the broadcast-zero linear correction preserve every scientific and
restart contract.  The job nevertheless failed the unchanged `1.10` memory
limit: R128 reserved memory was `1.180180x`, while R320 allocated and reserved
memory were `1.106109x` and `1.113065x` the homogeneous control.

## Attribution

The candidate no longer owns a dense lift, affine Laplacian, or spectral-zero
correction.  Its remaining lifting-owned dense allocation was the physical-Q
reconstruction workspace.  At R320 this five-component float64 field occupies
exactly `327,680,000` bytes.  Keeping it alive throughout the timestep is not
required by the homogeneous-remainder evolution or checkpoint contract.

The R128 reserved-memory failure is also sensitive to CUDA allocator size
classes, but the recovery does not relax or reinterpret either frozen memory
limit.  It first removes the remaining persistent full-domain lifting
workspace and lets the existing H100 analyzer adjudicate both allocated and
reserved memory again.

## Recovery

- The evolved `Fields` object remains the sole homogeneous-remainder state.
- Static lift values remain immutable wall-normal broadcast profiles.
- The complete-stress bulk-H, algebraic-stress, and Q-nonlinear pointwise
  kernels receive remainder and lift views separately.  Under the existing
  full-graph `torch.compile` policy, physical-Q addition belongs to the same
  pointwise graph instead of a persistent five-component workspace.
- Eager lift-aware kernels retain the exact reference operation order and are
  tested component by component against explicitly reconstructed physical Q.
- Physical observation reconstructs one requested component on demand.
- Gradient correction, linear correction, wall semantics, transform bases,
  transform counts, homogeneous-remainder checkpoint representation, runtime
  selection, and production defaults do not change.
- Storage metadata reports zero persistent physical-workspace bytes and keeps
  storage diagnostics outside scientific checkpoint identity.

On the local `8x8x6`, three-step compiled-runtime comparison, this recovery
and baseline `046ad5b` produced the same internal-final SHA-256
`d18743a084a95374d3fc92fb628ea81684b0e748122711e6f6b5abe9b2f07c3f`,
the same physical-final SHA-256
`9542397e9b3a9601f9ec8c2ff4e27853d71032102b437fecd48c2945ae1ddaf0`,
and the same `7/32` transform calls per step.

The original timestep and memory thresholds remain unchanged.  One new H100
recovery is required before P8.4.5 can close.  Nonhomogeneous Neumann data,
P8.5, Phase 9, compiled-runtime promotion, and production-default changes
remain outside scope.
