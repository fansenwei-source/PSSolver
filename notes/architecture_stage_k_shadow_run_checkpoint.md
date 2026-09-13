# Architecture Stage K: Plane shadow-run and checkpoint boundary

## Scope and result

Stage K places an opt-in run boundary around the Stage J coupled Plane
Beris--Edwards runtime.  It connects the separated model architecture to four
operational concerns that had deliberately remained in the production driver:

```text
external initial Q
        |
        v
projected coupled runtime -> synchronized Q/u/p observations
        |                              |
        v                              v
complete restart checkpoint      production-layout snapshots
        |
        v
separate continuation run
```

The checkpoint and observation products are intentionally different.  A
`Q/u/p` observation is suitable for analysis and uses the established Plane
filenames, but it is not presented as an exact integrator restart.  A complete
shadow checkpoint additionally preserves the evolved spectral state,
spectral-refresh phase, and algebraic solver warm-start state required to
continue the same floating-point trajectory.

This stage remains experimental and opt-in.  It does not replace
`Plane_beris_edwards_stokes.py`, alter Plane or Channel implementations, change
any numerical default, or modify the benchmark `develop` worktree.

## K1: strict external initial-condition injection

`ExperimentalPlaneShadowRun` accepts the five evolved Q components in their
declared order.  Every tensor must have the runtime's physical shape and exact
real dtype and must contain only finite values.  Missing components, additional
components, reordered components, implicit precision changes, and non-finite
values are rejected before a run begins.

The runtime's existing `reset()` lifecycle performs the qualified projection
of the injected state before the constitutive DAG is synchronized.  Metadata
records separate SHA-256 identities for the raw input Q and the projected Q,
plus caller-supplied JSON-compatible initial-condition provenance.  This keeps
initial-condition provenance explicit without embedding a V1 or V2 generator
inside the model or solver layer.

The output path is reserved before the runtime is reset.  A nonempty existing
directory is refused before any runtime mutation, so a new shadow run cannot
silently mix with or overwrite previous results.

## K2: synchronized production-layout observations

An observation first synchronizes the full algebraic DAG against the current
Q without advancing time or changing the spectral-refresh clock.  It then
exports:

```text
Q_<step>.npy : (Nx, Ny, Nz, 5)
u_<step>.npy : (Nx, Ny, Nz, 3)
p_<step>.npy : (Nx, Ny, Nz)
```

The arrays retain the runtime's real precision and contain finite values.  The
writer refuses existing frame files and uses same-directory temporary files
for atomic replacement of each array.  The existing snapshot loader can read
the result, so downstream analysis is not tied to the new runtime classes.

Completion synchronizes and validates the final observation, writes complete
metadata, and creates `COMPLETE` last.  If the final step was already saved,
the in-memory and on-disk dtype, shape, and values must agree exactly.  A
partial final observation is rejected.

## K3: complete exact-restart checkpoints

`ShadowRunCheckpoint` captures:

- every evolved Q component in physical space;
- the corresponding evolved Q component in native spectral storage;
- the spectral-refresh interval, phase counter, and refresh count;
- the completed global step count;
- algebraic implementation warm-start tensors with their dispatch identity;
- a SHA-256 identity of the complete trajectory-defining runtime contract.

Stored algebraic fields such as velocity and pressure are not checkpointed as
independent state because they are deterministic instantaneous functions of Q.
Transient constitutive fields are likewise not checkpointed.  Both are rebuilt
by a synchronized algebraic evaluation after the evolved state and solver warm
state are restored.

The runtime identity includes geometry, physical and spectral plans, model and
numerics metadata, real and spectral precision, timestep, and the effective
spectral-refresh interval.  A mismatch is a hard error.  Restore copies the
saved physical and spectral Q states directly; it does not round-trip one
through a transform and thereby perturb the restart state.

The on-disk representation is pickle-free.  Each tensor is stored in a separate
`.npy` file with shape, dtype, and SHA-256 records, while `checkpoint.json`
contains the frozen identities and counters.  A checkpoint is assembled in a
new sibling staging directory and atomically renamed into place.  Existing
checkpoint directories are never overwritten.  Loading verifies file identity,
shape, dtype, and finite values before constructing the checkpoint object.

## K4: production metadata comparison

The shadow path exposes a canonical scientific signature that can be compared
directly with metadata emitted by the existing production Plane driver.  The
comparison includes:

- physical and spectral shapes, domain lengths, timestep, and precision;
- transform execution order and spectral storage;
- canonical Q convention and all migrated Beris--Edwards/Stokes coefficients;
- Q, tangential velocity, normal velocity, and pressure boundary conditions;
- dealiasing, Hermitian axis, projected-transform mode, molecular-field
  assembly, pointwise execution, stress-divergence summation, and velocity
  zero-mode policy;
- the effective spectral-refresh interval in steps.

It returns all differing field paths and can promote any mismatch to a hard
error.  Qualification includes an actual `--dry-run` invocation of
`Plane_beris_edwards_stokes.py`, not only a synthetic copy of its schema.  The
matching shadow runtime and production metadata agree across the complete
currently migrated signature.

## K5: qualification

Stage K tests establish:

1. comparison with real production dry-run metadata and precise mismatch paths;
2. strict initial Q component, shape, dtype, finite-value, and output-directory
   gates;
3. raw/projected initial-state hashes and JSON-compatible run metadata;
4. synchronized production-layout Q/u/p snapshots readable by the established
   loader;
5. exact continuous-versus-restarted eight-step trajectories for both
   full-complex and Hermitian-half storage;
6. exact continuation across a spectral-refresh boundary, including phase and
   refresh counters;
7. hard runtime-identity and tensor-checksum failures;
8. continuation frame numbering and restart provenance;
9. completion ordering and refusal to mutate a completed run;
10. continued isolation from production Plane, Channel, solver, and top-level
    public APIs.

Desktop CPU qualification after the final implementation:

- architecture, metadata, snapshots, Beris--Edwards force/stress/energy, Fig.4
  CLI, and free-slip Stokes focus set: `209 passed, 3 subtests passed`;
- complete repository suite: `768 passed, 8 subtests passed`.

Both commands used an external Python bytecode cache and disabled the pytest
cache.  No GPU qualification is required because Stage K changes no production
path or numerical default.

## Production boundary after Stage K

Stage K proves that the separated Plane architecture can begin from an
externally supplied Q state, emit analysis-compatible observations, survive an
exact checkpoint/restart boundary, and demonstrate configuration parity with
the existing production driver.  It still does not claim long-trajectory or
H100 performance equivalence, and it does not promote the shadow runtime into
production.

The next stage should add a small explicit shadow entry point and execute a
side-by-side trajectory qualification against the existing Plane driver.  The
first gate should use identical initial Q, float64, the same refresh phase, and
short CPU runs; only after that succeeds should a bounded H100 equivalence and
performance test be designed.  Production selection should remain a later,
separate decision.

Channel constitutive parity and Channel-specific checkpoint or boundary
semantics remain outside this migration.  They must be introduced through
their own exact geometry registrations rather than inherited from Plane.
