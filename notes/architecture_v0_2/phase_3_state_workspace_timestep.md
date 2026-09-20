# Phase 3 plan: state, workspace, and timestep ownership

Status: `P3_4_SEPARATED_CANARY_H100_QUALIFIED`.

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

P3.0 through P3.4 are complete locally.  `pssolver.execution.state` and
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
[machine-readable record](phase_3_p34_h100_qualification.json).  P3.5 may now
connect the same contracts beneath the production compatibility facade; this
qualification does not promote `separated_canary` or change the production
default.
