# P8.6.2 integrated H100 closure plan

## Purpose

P8.6.2 is the single final-source H100 integration closure for Phase 8.  It
does not introduce a new runtime or repeat every historical performance study.
It proves that one installed wheel from one frozen source can select and run
all six qualified public runtime paths, preserve the already-qualified strong-Q
lifting path, and preserve the internal finite-Q Robin relaxation pilot.

The exact qualification source is supplied by the separate P8.6.2 source
binding record.  The binding must name a commit containing the P8.6.1 record,
this plan, the analyzer, and all tests, and must be the exact commit used to
build the qualification wheel.

## Preconditions

Before allocating a GPU, the HPCC task must:

1. verify the P8.6.1 classification and SHA-256;
2. create a detached clean worktree at the bound source commit;
3. run the full CPU suite while deselecting only the six known CUDA-only tests;
4. allow no skip except the already-known optional `nematics3d` dependency;
5. run the repository P8.6.1 analyzer and require its exact PASS classification;
6. build one wheel from the bound commit and import only that installed wheel;
7. run the eight-case structured rejection matrix before GPU allocation or as
   part of the CPU gate.

The six CUDA-only tests must run inside the one H100 job, before any trajectory,
and must all pass without skip, deselection, xfail, or failure.

## One-job H100 matrix

The same installed wheel and one H100 job must exercise:

1. Plane complete-stress, `legacy_production`;
2. Plane complete-stress, `compiled_v2`;
3. Channel legacy active-force, `legacy_channel`;
4. Channel legacy active-force, `compiled_channel_v2`;
5. periodic-box complete-stress, `periodic_spectral`;
6. rectangular-Channel complete-stress, `channel_complete_stress`;
7. Plane static prescribed-Q lifting, including homeotropic and planar conveniences;
8. the internal finite-Q Robin relaxation pilot.

Each public runtime path must record requested/effective runtime identity,
adapter identity, no fallback, finite state, concrete CUDA device identity,
Q/u/p output identity where the application owns hydrodynamic fields, and
continuous-versus-split/restart equality under its existing qualified
contract.  The lifting and finite-Q cases retain their own already-qualified
state and checkpoint contracts; finite-Q remains an internal partial pilot and
must not appear in the public capability catalog.

## Gates

The final analyzer must fail closed unless:

- all six runtime paths are present exactly once in the declared matrix;
- all runtime requests equal their effective paths and `fallback_used=false`;
- every state and output array is finite;
- continuous and split/restart results meet their existing exact or tolerance
  contracts;
- all checkpoint identity, payload, file-record, runtime, and backend negative
  gates reject before target mutation;
- Plane and Channel production defaults remain `legacy_production` and
  `legacy_channel`;
- Robin remains public declaration-only;
- no source checkout shadows the installed wheel;
- the worktree remains detached and clean;
- no new performance, paper-benchmark, or steady-state claim is made.

P8.6.2 is a correctness and integration closure.  It has no paired speedup
threshold.  Runtime and memory measurements are provenance, not promotion
criteria, unless a catastrophic OOM or non-finite condition occurs.

## Stop and retry policy

Exactly one formal H100 job is authorized after the source binding exists.
Requeue and automatic retry are disabled.  A pre-script scheduler bootstrap
failure may be reported separately, but no second submission is implied by
this plan.  Any scientific, package, identity, finite-state, restart, or
negative-gate failure stops the task without entering P8.6.3.

P8.6.3 and Phase 9 remain unauthorized until a complete, checksummed P8.6.2
archive has been independently reviewed and recorded.
