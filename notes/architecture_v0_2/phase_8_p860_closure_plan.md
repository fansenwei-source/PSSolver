# Phase 8 P8.6.0: cumulative closure plan

Status: `PASS_P8_6_0_PLANNING_CUMULATIVE_CLOSURE_NOT_EXECUTED`.

Baseline: `2c100b88701e1beadc8eddc73d0a313fb96d7f18`, after the
authoritative P8.5.6 H100 closure.  P8.6 freezes and audits the integrated
Phase 8 capability surface; it adds no equation, boundary law, geometry,
runtime path, numerical kernel, or production default.

## Prerequisite evidence

The cumulative closure imports five immutable authorities:

1. P8.1 public declarations and catalog;
2. P8.2 periodic complete-stress H100 closure;
3. P8.3 rectangular-Channel complete-stress H100 closure;
4. P8.4.5 Plane static-Dirichlet lifting H100 closure;
5. P8.5.6 finite-Q Robin relaxation H100 closure.

Every record is bound by repository path and SHA-256.  Failed or superseded
recovery records remain historical provenance but are not scientific closure
authorities.  P8.6 must not reinterpret their partial measurements as a new
qualification result.

## Frozen public capability catalog

The public catalog at closure contains exactly:

- two model declarations: complete-stress Beris--Edwards and the legacy
  active-force Channel model;
- three tensor-product geometries: periodic box, Plane slab, and rectangular
  Channel;
- eight boundary declarations: Robin, homogeneous Q Neumann, free-slip
  velocity, no-slip velocity, pressure compatibility, prescribed Q, strong
  homeotropic Q, and strong planar Q;
- four executable model--geometry combinations.

The four combinations are:

```text
complete_stress_beris_edwards + plane_slab
  -> plane_complete_stress_beris_edwards
  -> legacy_production | compiled_v2

legacy_active_force_active_nematics + rectangular_channel
  -> channel_legacy_active_force_active_nematics
  -> legacy_channel | compiled_channel_v2

complete_stress_beris_edwards + periodic_box
  -> periodic_complete_stress_beris_edwards
  -> periodic_spectral

complete_stress_beris_edwards + rectangular_channel
  -> channel_complete_stress_beris_edwards
  -> channel_complete_stress
```

Declaration is not execution.  `robin` remains field-neutral and publicly
declarable, but it has no qualified public application.  P8.5 qualified only
an internal five-component wall-normal relaxation pilot, not a complete
Beris--Edwards/Stokes public timestep.  The catalog must therefore continue
to report Robin as non-executable.

Strong prescribed-Q policies are executable only through the qualified Plane
application.  A periodic direction is geometry-induced and is not a wall
policy.  No declaration may claim compatibility with an application that has
not passed lowering, runtime, restart, and H100 gates.

## Cumulative rejection matrix

Compilation must fail before allocation and without fallback for:

- any unregistered model--geometry pair;
- Robin Q through the public complete timestep;
- prescribed or strong Q on an unqualified geometry;
- a wall policy on a periodic face;
- velocity or pressure policies incompatible with the selected geometry;
- nonzero prescribed Neumann flux;
- time-dependent, spatially varying, callable, or trainable Robin data;
- checkpoint runtime, backend, geometry, boundary, lifting, surface-law,
  tensor, file-record, or progress identity mismatch.

Each rejection must use its existing structured code at the layer that owns
the failure.  P8.6 does not introduce one catch-all error and does not turn an
unsupported combination into a fallback.

## P8.6 execution slices

### P8.6.1 — local cumulative audit

Add a repository-owned fail-closed analyzer and tests that:

- hash-bind all five authoritative Phase 8 records;
- compare the immutable public catalog with the live compiler registry;
- freeze public imports and declaration metadata;
- audit application/runtime/default ownership;
- execute the structured rejection matrix before allocation;
- verify current-source CPU manufactured, restart, and tamper controls;
- build and import a wheel outside the checkout;
- produce a machine-readable H100 command plan without submitting it.

P8.6.1 may add only tests, audit tooling, and records.  It must not change a
runtime, numerical operator, compiler registration, declaration, checkpoint
schema, or public API.

### P8.6.2 — single-H100 integrated closure

Use one installed wheel from the frozen P8.6.1 source and exactly one H100
job.  Before scientific commands, run every CUDA-only repository test that
the login-node CPU gate precisely transfers to the device gate.

The H100 job then runs a small current-source smoke for all six public runtime
paths, plus the qualified strong-Dirichlet lifting path and finite-Q Robin
relaxation pilot.  The job must prove finite state, requested/effective
runtime identity, no fallback, expected transform/linear-solver counts,
checkpoint/restart identity where the runtime supports restart, and bounded
live memory.  Matched legacy/compiled Plane and Channel pairs must retain
their existing numerical-equivalence contracts.

This is an integration non-regression closure, not a new performance contest.
The authoritative detailed performance results remain the slice-specific H100
records.  P8.6.2 uses conservative current-source smoke thresholds only to
detect gross integration regressions, foreign-process contamination, OOM,
fallback, or a changed execution topology.

### P8.6.3 — final Phase 8 record

After P8.6.1 and P8.6.2 pass, create `phase_8_final_closure` records binding
the complete catalog, rejection matrix, local audit, installed-wheel result,
single-H100 archive, defaults, and non-claims.  `COMPLETE` is written only by
the H100 evidence bundle after every gate passes.

## Compatibility and non-claims

Plane and Channel defaults remain `legacy_production` and `legacy_channel`.
Neither compiled runtime is promoted.  Periodic and complete-stress Channel
paths remain their explicitly selected single runtimes.  P8.6 changes no
checkpoint or output format and makes no Shendruk, Minu, stationarity,
long-time, or paper-reproduction claim.

P8.6 closure will not mean that arbitrary combinations are executable.  It
will mean the finite registered set is explicit, package-installable,
fail-closed, restart-audited, and collectively non-regressed on the final
Phase 8 source.

## Authorization boundary

This record authorizes P8.6.1 local cumulative-audit implementation only.
It does not authorize the P8.6.2 H100 submission, Phase 9, optimal control,
new boundary physics, nonhomogeneous Neumann work, production-default changes,
or compiled-runtime promotion.  A passing P8.6 final closure may make Phase 9
planning eligible; Phase 9 implementation still requires separate authority.

The planning audit passed five targeted tests and the complete local suite of
`2403 passed, 8 subtests passed`, with no failure, skip, deselection, or xfail.
The existing verbatim architecture PDF was not regenerated or modified.
