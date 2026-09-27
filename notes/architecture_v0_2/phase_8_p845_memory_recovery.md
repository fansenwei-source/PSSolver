# P8.4.5 static-lifting memory recovery

## Status

`READY_P8_4_5_STATIC_LIFTING_MEMORY_RECOVERY`

The CUDA allocated-device recovery was scientifically successful, but Job
10843551 failed the frozen memory ceiling.  All numerical, wall, manufactured,
restart, transform-count, graph-break, fallback, and timestep gates passed.
Peak allocated/reserved ratios were 1.247/1.142 at R128 and 1.266/1.230 at
R320, above the frozen 1.10 limits.

## Attribution

At R320 one dense five-component float64 field occupies 327,680,000 bytes.
The observed allocated-memory delta was approximately five such fields.  The
old representation retained dense full-domain lift values, dense explicit
zero affine Laplacians, a dense physical reconstruction workspace, a dense
zero spectral linear correction, and the qualification caller's physical
initial values in addition to the evolved homogeneous remainder.

This is the duplicate-domain-storage failure that the original 10% gate was
designed to detect.  The gate is not relaxed.

## Recovery

- Affine lifts are stored as component-by-wall-normal profiles and exposed as
  zero-stride logical full-domain views.
- Explicit zero affine Laplacians use the same compact profile layout.
- When the model declares the linear lift correction identically zero
  (`ldg_a == 0` for this frozen case), the spectral correction is a broadcast
  complex scalar rather than a dense spectral tensor.
- The qualification helper releases caller-owned construction inputs and the
  CUDA allocator's unused construction cache before warmup.  It does not
  clear the cache between warmup and the timed window.
- The preallocated physical reconstruction workspace remains dense and is
  still the only full-domain lifting-owned working field in the hot path.
- Restart metadata now reports logical extent separately from owned storage.

The timestep equations, homogeneous-remainder representation, transform
bases, transform counts, operation order, public boundary semantics, and
frozen memory/timestep thresholds are unchanged.  A CPU comparison against
commit `0e4a5fb` produced identical initial, internal-final, and
physical-final SHA-256 identities, as well as identical lifting checkpoint
metadata.

One new H100 recovery is required.  Nonhomogeneous Neumann data, P8.5, Phase
9, compiled-runtime promotion, and production-default changes remain outside
scope.
