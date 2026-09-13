# Architecture Stage L: production-versus-shadow Plane trajectories

## Scope and result

Stage L turns the Stage K run boundary into an executable, side-by-side CPU
qualification path.  The existing production driver remains the reference and
the migrated architecture remains an opt-in shadow:

```text
completed production CPU run
       | metadata.json + projected Q_0
       v
validated reference adapter
       | exact physical/numerical configuration
       v
experimental Plane runtime -> shadow Q/u/p trajectory
       |                              |
       +------ read-only comparator --+
                         |
                         v
                 PASS / FAIL bundle
```

The new entry point does not duplicate the production driver's full argument
parser or benchmark parameterization.  It reconstructs the migrated runtime
from the authoritative metadata of one completed reference run and injects
that run's saved projected `Q_0.npy`.  This prevents a second independently
maintained set of activity, Frank-to-LdG, boundary, precision, transform, or
refresh rules from drifting away from the benchmark path.

Stage L changes no production entry point, Plane or Channel solver, numerical
default, or benchmark `develop` worktree.

## L1: a deliberately narrow reference contract

`load_production_plane_reference()` accepts only a run that satisfies the
first migration gate:

- script identity is `Plane_beris_edwards_stokes.py`;
- active-nematic model identity is the complete-stress Beris--Edwards/Stokes
  variant;
- metadata and the `COMPLETE` marker both report successful completion;
- execution was CPU and float64, with TF32 ineffective;
- snapshots begin at step zero and include Q, velocity, and pressure;
- production Q-gradient reuse is enabled;
- the saved `Q_0.npy` shape, dtype, and finite-value checks pass;
- the ordered Q identity, including the production batch dimension, agrees
  with the projected initial-Q SHA-256 recorded in metadata.

The first gate is intentionally CPU-only.  It isolates architectural and
equation equivalence from GPU compilation, scheduling, and performance.  A
future H100 gate must be a separate stage with its own authorization and
provenance.

The shadow command requires `--confirm-steps` to equal the completed reference
step count.  This prevents accidentally pointing the qualification tool at a
large production run and silently starting an equally large replay.

## L2: metadata-driven runtime reconstruction

`build_plane_shadow_runtime_from_production_metadata()` reconstructs:

- the Plane domain, physical shape, spectral shape, and timestep;
- Q, tangential velocity, normal velocity, and pressure boundary sets;
- all migrated Beris--Edwards and Stokes coefficients;
- the tangential zero-mode policy and friction;
- precision, dealiasing, transform execution, native spectral storage, and
  Hermitian axis;
- molecular-field, pointwise-kernel, and stress-divergence policies;
- pressure diagnostics and the effective spectral-refresh interval.

Construction still goes through the typed model, geometry, numerical-policy,
and exact Plane solver registries introduced in Stages A--J.  No production
solver object is imported into the shadow model.  After construction, the
Stage K scientific-signature comparison must be exactly compatible before an
output directory is created or a trajectory begins.

The explicit entry point is:

```text
Plane_beris_edwards_shadow.py
    --production-reference-dir <completed-reference>
    --output-dir <new-empty-directory>
    --confirm-steps <exact-reference-step-count>
```

It saves step zero, advances one timestep at a time, follows the reference
save interval, synchronizes algebraic Q/u/p observations, and writes the final
metadata and `COMPLETE` marker through the Stage K run coordinator.  Source
metadata and initial-Q file SHA-256 values are embedded in the shadow metadata.

The injected array is exactly the saved production physical `Q_0`.  Because
the shadow runtime must construct its own native spectral state, it performs
the qualified projected transform once more.  Cross-architecture comparison
therefore uses a floating-point tolerance rather than claiming identical
hidden spectral coefficients that the production run did not save.

## L3: read-only scientific comparison

`compare_plane_shadow_trajectories()` verifies before reporting a result:

1. both runs have complete metadata and valid completion markers;
2. the saved shadow and production scientific signatures match;
3. the source production metadata and `Q_0` SHA-256 values have not changed;
4. completed step counts agree;
5. shadow Q/u/p frame sets agree and every corresponding production frame
   exists;
6. every paired array has identical shape and dtype and contains only finite
   values.

Metadata, initial Q, and every compared array are hashed before and after they
are read.  A concurrent or accidental input change is therefore fatal instead
of producing a report whose recorded identity differs from the analyzed bytes.

For Q and velocity the gate is raw relative L2 error.  For pressure it is the
demeaned relative L2 error, so an irrelevant constant pressure gauge cannot
cause a false failure.  Raw pressure error, both Linf errors, file SHA-256
values, and byte-identity status remain in the report.  The initial Stage L
tolerance is `1e-10`.

The comparator never writes into either simulation directory.  Its separate,
new output directory contains `comparison.json` followed by exactly one
terminal marker: `COMPLETE` for PASS or `FAILED` for FAIL.  It refuses a
nonempty output directory.

## L4: real CLI-to-CLI qualification

The tests invoke the actual production script, the actual shadow entry point,
and the standalone comparator.  Both qualified numerical paths are covered:

```text
Plane default:          hermitian_half + truncated
validated rollback:    full_complex   + full
```

Each reference and shadow run uses float64, saves every step from 0 through 6,
and uses a spectral-refresh interval of two steps.  Thus the comparison spans
three refresh boundaries rather than testing only the initial pre-refresh
phase.  Each configuration compares 21 arrays: seven frames times Q, u, and p.

Observed maximum gate relative L2 errors were:

- `1.4577486971639142e-15` for Hermitian-half/truncated;
- `1.2307838829701740e-15` for full-complex/full.

Both are far below `1e-10` and also below `1e-12`.  A controlled `1e-5`
perturbation to one final Q element produces a comparison failure and a
`FAILED` bundle, confirming that the gate is active rather than merely
descriptive.  The tests also prove reference-directory read-only behavior,
step-confirmation refusal, unsupported Q-gradient policy rejection, CPU-only
enforcement, and continued production isolation.

Desktop CPU qualification after the final implementation:

- architecture, metadata, snapshots, Beris--Edwards force/stress/energy,
  Fig.4 CLI, and free-slip Stokes focus set:
  `216 passed, 3 subtests passed`;
- complete repository suite: `775 passed, 8 subtests passed`.

All commands used an external Python bytecode cache and disabled the pytest
cache.  No HPCC job was needed for Stage L.

## Production boundary after Stage L

Stage L establishes equation- and trajectory-level CPU parity between the
existing production Plane driver and the separated architecture for a short
run that crosses repeated spectral refreshes.  It does not establish H100
performance, long-time chaotic trajectory identity, statistical benchmark
equivalence, or production readiness.

The next stage should be a bounded H100 gate using the same reference-driven
workflow.  It should first run a small preflight, then compare production and
shadow numerical outputs under a relative-L2 scientific tolerance and measure
runtime and peak memory separately.  It must not replace the production path
or start a long benchmark.  Default selection should remain a later decision
after GPU correctness and performance evidence are both available.

Channel remains outside this migration.  Its constitutive boundary spaces,
geometry solver registrations, and performance qualification require a
separate path rather than inheriting Plane assumptions.
