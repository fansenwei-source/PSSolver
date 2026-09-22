# Phase 5 P5.5: opt-in compiled-v2 application connection

Status: `PASS_P5_5_OPT_IN_APPLICATION_CONNECTION`.

Baseline: P5.4 commit `1da0ceaf7b4af159d9ce363dd51476bc85e70d52`.

## Scope

P5.5 connects the already-qualified P5.1--P5.4 compiled Plane pieces to the
supported application boundary.  It adds the explicit runtime identity
`compiled_v2`, parser support, a private production-facing runtime adapter,
and exact factory dispatch.  It does not change equations, boundary
conditions, time integration, tensor layout, checkpoint format, output
format, or numerical kernels.

The selector remains additive:

- an omitted `--runtime-path` still resolves to `legacy_production`;
- `separated_canary` remains an explicit diagnostic path;
- `compiled_v2` is an explicit opt-in path;
- requested and effective identities are identical;
- construction or execution failure is propagated and never falls back.

## Runtime assembly

The compiled path uses the package-owned Plane production assembly to create
the already-qualified tensors, transforms, projector, Beris--Edwards kernels,
and free-slip Stokes kernel.  It then performs the P5.2 construction audit,
builds the P5.3 fixed Euler step program, and composes the private P5.4
observation/checkpoint adapter behind `PlaneRuntimeAdapterProtocol`.

The hot timestep delegates directly to the pre-bound program.  Runtime
selection, metadata, construction validation, observation, serialization,
and restart validation remain outside that timestep.

## Checkpoint identity

Checkpoint format remains version 1 and the tensor/file schema is unchanged.
Now that the selector exists, compiled checkpoints carry the actual
`compiled_v2` runtime path rather than the temporary P5.4
`legacy_production` carrier.  Cross-runtime restart is rejected before tensor
mutation or stepping.  Compiled backend identity and the run's runtime
identity SHA-256 remain mandatory.

## Dry-run and default behavior

The application returns from dry-run before output-directory creation,
initial-condition generation, runtime assembly, or tensor allocation.  The
dry-run metadata records `compiled_v2` only when explicitly requested.

The production default remains `legacy_production`.  P5.5 does not promote
either opt-in path and does not authorize Phase 6 or a default change.

## Local validation

The P5.1--P5.5 targeted suite passes with no failure or skip.  It verifies:

- exact three-value selector membership and the unchanged default;
- explicit parser resolution of `compiled_v2`;
- private adapter construction and callback semantics;
- failure propagation without fallback;
- tensor-free, output-free compiled dry-run;
- short compiled/legacy Q/u/p byte identity;
- compiled continuous/split checkpoint-restart byte identity;
- actual compiled checkpoint runtime identity;
- P5.1--P5.4 private surfaces remain absent from package exports.

The complete CPU suite also passes.  The authoritative counts are stored in
`phase_5_p55_application_connection.json`.

## Next boundary

P5.6 is locally eligible.  It owns the installed-wheel workflow smoke, local
CUDA smoke when available, deterministic 1/2/100-step closure, negative
identity gates, tensor-lifetime audit, hot-loop static audit, and the frozen
Phase 6 H100 qualification handoff.  P5.6 is not implemented by this record.
