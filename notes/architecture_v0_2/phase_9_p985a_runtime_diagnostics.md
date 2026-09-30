# P9.8.5a stable functional runtime diagnostics

Status: `PASS_P9_8_5A_LOCAL_STABLE_RUNTIME_DIAGNOSTICS`

The P9.8.5 recovery exposed one genuine public-surface gap: Channel pressure
convergence was already represented by a stable JSON schema, but the stable
runtime facade offered no public route to the actual primal and transpose
outcomes.  The external qualification helper therefore attempted to reach a
private `_flow_model` member and correctly failed.

PSSolver commit `0ee4426` adds `diagnostics()` to every concrete functional
runtime and to the stable factory facade.  The facade serializes with
`allow_nan=False` and deserializes before returning, so callers receive a
fresh, finite JSON object rather than a reference to mutable implementation
state.  Non-JSON objects and NaN/Inf are rejected at the public boundary.

Periodic execution reports `pressure: null`, because it uses a direct Fourier
pressure operation.  Channel execution reports the latest primal and
conjugate-transpose pressure results using the already qualified convergence
schema.  Both begin with an explicit `not_run` state; after execution they
include requested and achieved iterations, absolute and relative residuals,
tolerance semantics, termination reason, and the final `acceptable` decision.

The API 1.0 `FunctionalRuntimeProtocol` is not expanded.  Making an additive
diagnostics method a new required structural member would reject proxy
implementations that already conform to the frozen 1.0 protocol.  Instead,
runtime objects returned by the stable factories guarantee this additive
method.  A future protocol minor version may make the extension structural.
This choice preserves all earlier consumer and source-identity records.

## Local evidence

- PSSolver targeted diagnostics tests: `3 passed`;
- PSSolver implementation commit suite: `2614 passed, 8 subtests passed`;
- PSSolver record-tree suite: `2615 passed, 8 subtests passed`;
- PSSolver-Control complete suite: `239 passed, 1 skipped`, with only the
  existing opt-in R5 CUDA benchmark skipped;
- external installed-wheel diagnostics test: `1 passed`;
- installed imports resolve from the fresh venv's `site-packages`, not either
  source checkout.

The PSSolver and PSSolver-Control wheel SHA-256 values are respectively
`634399ba409183d585312a4bd8a4cc1bfff5c27afa258321c0451eb37184979c`
and `aacc3fc2263d320cfa6e2fdf72aed14d4ce895908aa5dfc6d1aee195a3382f11`.

No runtime identity, checkpoint schema, equation, transform, pressure
algorithm, numerical default, or dependency changed.  The shared environment
and `nematics3d` were not modified.  Historical P9.8 evidence was not
rewritten, and the cumulative PDF was not regenerated.

P9.8.5a authorizes a narrowly scoped P9.8.5 Channel recovery using the public
diagnostics method.  It does not itself complete P9.8.5 or Phase 9.
