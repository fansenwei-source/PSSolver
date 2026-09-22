# Phase 3 plan: state, workspace, and timestep ownership

Status: `PASS_PHASE3_CLOSURE_NON_REGRESSION_WITH_AUTHORIZED_COMPOSITE_RECOVERY_V7`.

Baseline: Phase 2 closure commit
`0d5b7186b7a104354db173e04a43e68c15daa515` on
`next/pssolver-v0.2.0-architecture`.

The normative ownership decision is recorded in
[ADR 0009](adr/0009-runtime-state-workspace-and-step-program.md).  The
machine-readable audit is
[phase_3_state_inventory.json](phase_3_state_inventory.json).

## Objective

Separate mutable evolved state, generation-local work, and fixed timestep
execution without changing the qualified Plane equations or hot path.  The
end state has one `RuntimeState`, a bounded runtime workspace, and a pre-bound
step program shared by compatible integrator facades.

## Audited current ownership

| Current owner | Persistent data | Reconstructible/work data | Migration target |
|---|---|---|---|
| `Fields` | evolved physical and spectral Q | u, p, transform operations and metadata | evolved arrays to `RuntimeState`; layout/resources to execution plan; algebraic arrays to workspace |
| integrator | refresh counters | current-static-field flag | counters to `RuntimeState`; validity flag to workspace/program state |
| `PDEModel` | parameters and declarations | nonlinear/static execution dispatch | immutable execution/model plan |
| projector/backend | masks and transform resources | transform intermediates | execution plan plus bounded workspace |
| algebraic adapters | authorized persistent solver restart tensors | gradients, H, stress, force, flow and representation cache | persistent tensors to `RuntimeState`; all other values to workspace |
| workflow checkpoint | cloned evolved Q and refresh phase | none | format-v1 adapter over `RuntimeState` during migration |

The current `completed_steps` value is derived from integrator refresh
counters.  Phase 3 makes it explicit and requires the phase counters to remain
mathematically consistent with it.

## Frozen non-change contract

- production default remains `legacy_production`;
- `separated_canary` remains explicit and non-promoted;
- checkpoint format version remains 1 through the first connection;
- supported public imports, signatures, pickle behavior, metadata schema, and
  output schedule remain unchanged;
- the projected Plane timestep follows the nine-operation oracle in ADR 0009;
- no tensor clone, conversion, or new allocation is permitted when a legacy
  storage view is adopted by `RuntimeState`;
- no metadata construction, registry lookup, string dependency discovery, or
  checkpoint work enters the hot loop.

## Phase slices

### P3.0: audit and freeze

Freeze the ownership inventory, timestep order, checkpoint contract, public
compatibility boundary, numerical oracle, and performance gates.  This slice
changes documentation and tests only.

### P3.1: disconnected `RuntimeState`

Introduce explicit evolved physical/spectral storage, an integrator-progress
clock, a representation-generation ledger, and declared persistent algebraic
state.  The object stores supplied tensors by identity and is not connected to
either Plane runtime.

### P3.2: bounded workspace

Introduce an immutable workspace plan and preallocated runtime workspace.
Generation-local values receive validity tokens; diagnostics may copy or
summarize them but may not retain tensor references across generations.

### P3.3: frozen step program

Extract the common projected semi-implicit Euler sequence behind compatibility
facades.  Characterization tests must prove exact callback timing, refresh
phase, mutation order, exception behavior, and tensor identity.

### P3.4: canary connection

Connect state, workspace, and the step program only to `separated_canary`.
Continuous and split/restart trajectories must match its pre-Phase-3 oracle.
The legacy production path remains untouched and supplies rollback comparison.

### P3.5: production facade connection

After canary qualification, connect the same internal contracts beneath
legacy facades without changing stable imports or attributes.  Checkpoint v1
must restart bidirectionally across the pre-connection and post-connection
implementation for the same runtime path.

### P3.6: closure qualification

Run complete CPU suites, build/install tests, format-v1 checkpoint and tamper
gates, continuous/split trajectories, R128/R320 H100 balanced performance,
peak-memory, transform-call, fallback, and finite-value gates.  Promotion or
default changes require a separate explicit decision.

## Current implementation boundary

P3.0 through P3.5 are complete and independently qualified.
`pssolver.execution.state` and
`pssolver.execution.workspace` are provisional direct-import modules and are
deliberately absent from `pssolver.execution.__all__` and `pssolver.__all__`.
Neither runtime imports them.  Workspace buffers are allocated once from an
immutable bounded plan and guarded by generation tokens.  The provisional
`pssolver.integrators.step_program` freezes the qualified projected-IMEX
operation sequence over these contracts.  The explicit `separated_canary`
construction edge adopts its existing evolved arrays by storage identity and
uses the state-backed program.  Generic experimental builders and
`legacy_production` retain their previous integrators.  P3.4 passed its
independent R128/R320 H100 non-regression, continuous-trajectory,
bidirectional cross-version restart, checkpoint-tamper, and
cross-runtime-rejection gates.  The qualification is summarized in
[phase_3_p34_h100_qualification.md](phase_3_p34_h100_qualification.md) and its
[machine-readable record](phase_3_p34_h100_qualification.json).  P3.5 connects
the same state-backed core beneath the stable
`DealiasedSemiImplicitEulerIntegrator` production facade.  The facade name,
runtime selection, metadata surface, checkpoint format, model-specific inverse
storage operation, and default remain unchanged.  It passed independent CPU
and H100 non-regression qualification, including
legacy and canary continuous trajectories, bidirectional format-v1 restart,
reset/rebind, tamper rejection, and R128/R320 performance and memory gates.
The qualification is summarized in
[phase_3_p35_h100_qualification.md](phase_3_p35_h100_qualification.md) and its
[machine-readable record](phase_3_p35_h100_qualification.json).  P3.5 is
closed and P3.6 closure qualification is authorized; no runtime was promoted
and the production default remains unchanged.  The cumulative P3.6 local gate
also passes against the Phase 2 closure baseline: complete CPU suites, clean
sdist/wheel installation, both-runtime continuous trajectories, and
bidirectional cross-version restart remain exact.  Its
[local qualification record](phase_3_p36_local_qualification.md) and
[machine-readable contract](phase_3_p36_local_qualification.json) froze the
cumulative closure boundary.  The independent, pre-registered production
workflow simulation-timer adjudication subsequently passed at R128 and R320;
its [qualification summary](phase_3_p36_performance_adjudication.md) and
[machine-readable evidence](phase_3_p36_performance_adjudication.json) resolve
the performance non-regression gate without changing either runtime or the
default.  The subsequent cumulative closure continuation passed all eight
R128/R320 parent/candidate continuous trajectories and all four R128
bidirectional cross-version restart trajectories.  Its
[continuation record](phase_3_p36_closure_continuation.md) and
[machine-readable evidence](phase_3_p36_closure_continuation.json) preserve
those results.  The job stopped only because the external reset/rebind harness
supplied incomplete hand-built production metadata to the separated-canary
constructor.  Phase 3 remains open solely for a corrected reset/rebind smoke
and the eleven negative closure gates.  The first reset/rebind recovery then
confirmed a second harness-only problem: it released old objects before
comparing integer IDs and CUDA addresses, and it reused mutable initial tensor
objects across the evolving, reset, and fresh-control roles.  The
[reset/rebind recovery record](phase_3_p36_reset_rebind_recovery.md) and its
[machine-readable evidence](phase_3_p36_reset_rebind_recovery.json) freeze the
required correction: retain live old-object references and build independent,
non-overlapping initial clones.  No previously qualified trajectory, restart,
or performance gate needs to be repeated.  Recovery v3 then passed all
immediate reset/rebind contracts but raised an aggregate legacy post-reset
evolution error before persisting its individual predicates.  A local CPU
diagnostic reproduced the mismatch when progress restoration marked stale
fresh-control static fields current; following the production checkpoint
order (`synchronize_for_observation()` before `restore_progress()`) restored
bitwise identity.  The
[v3 diagnostic](phase_3_p36_reset_rebind_recovery_v3.md) and its
[machine-readable record](phase_3_p36_reset_rebind_recovery_v3.json) require
atomic subgate persistence and the production restoration order in the next
recovery.  Recovery v4 applied that order and passed all structural reset,
workspace, progress, and representation gates.  Its remaining independent-
reconstruction differences were only float64 roundoff (at most about
`4.6e-16`) while the deliberately unsynchronized control differed by as much
as `2.7e-4`.  The
[v4 adjudication](phase_3_p36_reset_rebind_recovery_v4.md) and
[machine-readable record](phase_3_p36_reset_rebind_recovery_v4.json) preserve
the distinction between workflow byte-identity oracles, which remain frozen,
and the reset-versus-independent-construction diagnostic, which now requires
a pre-registered strict float64 tolerance.
Recovery v5 implemented that tolerance but stopped before formal CPU trials
because an unsynchronized negative control produced the mathematically
infinite relative L2 associated with a zero-norm reference and nonzero
difference.  Strict JSON correctly rejected the non-standard infinity token.
The [v5 schema record](phase_3_p36_reset_rebind_recovery_v5.md) and its
[machine-readable contract](phase_3_p36_reset_rebind_recovery_v5.json) require
JSON `null` plus an explicit zero-reference status; the finite elementwise
violation count remains the qualification authority.
Recovery v6 completed that schema fix and passed all three legacy CPU trials.
It stopped only because the harness required an unsynchronized numerical
control to fail for the separated canary, although both the local diagnostic
and v6 show that this runtime has no observable dependency on that omitted
operation.  The [v6 applicability record](phase_3_p36_reset_rebind_recovery_v6.md)
and its [machine-readable contract](phase_3_p36_reset_rebind_recovery_v6.json)
retain numerical discrimination for legacy while requiring exact call-order
instrumentation and production-order tolerance for canary.

Recovery v7 applied that runtime-specific contract and completed the composite
P3.6 closure.  The separated-canary CPU trials, both-runtime H100 reset/rebind
checks, exact production event traces, and all eleven checkpoint negative
gates passed.  Together with the frozen performance, continuous-trajectory,
and bidirectional-restart evidence, this closes Phase 3 without changing the
production default or promoting the canary.  The
[final closure record](phase_3_p36_final_closure.md) and its
[machine-readable evidence](phase_3_p36_final_closure.json) authorize Phase 4
planning only; Phase 4 implementation remains separately unauthorized.
