# Bounded-axis execution architecture

## Scope

This branch starts from the immutable PSSolver `v0.1.1` release.  It separates
the device-bound execution of one DCT-II or DST-II axis from tensor-product
ordering without changing the transform algorithm.  Dense matrix execution
remains the only implementation and the production default.

The change deliberately does not add:

- an FFT or pruned real-to-real implementation;
- a runtime or per-call automatic selector;
- a new CLI or public solver option;
- a change to basis functions, normalization, mode order, boundary semantics,
  projection, dealiasing, storage, or PDE equations.

## Plan layers

`ComponentTransformPlan` remains the tensor-free mathematical description of a
field component.  It records the transform kinds, retained mode counts and
computed axis sizes required by a problem.

`BoundedAxisExecutionPlan` is a lower-level device-bound protocol for one DCT
or DST axis.  Its static identity is:

```text
(kind, physical_size, retained_count, device, real_dtype, value_type)
```

`TensorProductTransformBackend` remains the public compatibility facade.  It
owns transform order, periodic grouping, Hermitian storage, axis movement and
spectral metadata.  It caches bounded-axis plans and delegates only the final
axis matrix operation to the selected executor.

## Frozen dense executor

`DenseBoundedAxisExecutionPlan` preserves the `v0.1.1` operation:

```text
matrix construction -> retained-row view -> movedim -> matmul -> movedim back
```

The complete orthonormal matrix remains shared in the historical matrix cache.
A retained plan stores only a view into that matrix, so the abstraction does
not introduce a second matrix allocation.

The former private methods remain thin compatibility wrappers because existing
tests and profiling tools observe them.  The new implementation file is added
to Plane implementation provenance; scientific configuration remains fixed,
while the implementation identity changes audibly as it should.

The dense compatibility executor retains the historical ability to convert
its basis matrix to a non-canonical input tensor device or dtype at the final
matmul boundary.  Production solver tensors already match their backend.  A
future executor that owns device-specific twiddles or workspace must either
enforce that canonical-input contract or extend its plan identity with the
actual execution device and dtype; it must not reuse the present compatibility
behavior implicitly.

## Qualification boundary

This stage is acceptable only if it preserves byte-identical dense transform
results and Plane trajectories, passes the complete CPU suite and package
installation checks, and does not cause a stable GPU timestep or memory
regression.  A later experimental branch may add an explicit full-FFT executor
behind the same protocol.  A genuinely pruned executor remains conditional on
a measured low-retained-fraction workload and separate numerical and GPU
qualification.

## Stage-one qualification

The first implementation was compared against `v0.1.1` at commit `07bbf80`.
It passed the following local gates:

- all dense DCT/DST forward and inverse reference cases were bitwise equal for
  full and retained transforms, float32 and float64, real and complex values,
  and every tested tensor axis;
- the final Q, u and p files from a 100-step Plane run were byte-for-byte
  identical to the release baseline;
- the complete CPU suite passed with `1095 passed, 8 subtests passed`;
- a wheel and source distribution contained the backend package and this note,
  and a clean wheel install imported the executor and exposed the production
  console entry point;
- on an RTX 3060 Ti, three paired R128 runs measured 27.523 ms/step for the
  release and 27.412 ms/step for the candidate (candidate/release mean ratio
  0.9960), with identical peak memory and final-state hashes;
- on the same GPU, three paired R320 runs measured 712.978 ms/step for the
  release and 714.101 ms/step for the candidate (ratio 1.0016), with unchanged
  peak reserved memory, slightly lower peak allocated memory and identical
  final-state hashes.

The R320 change is a 0.16% timing difference, well inside the two-percent
non-regression gate.  This stage therefore qualifies the architecture seam,
not a faster DCT/DST algorithm.  Dense remains the sole implementation.
