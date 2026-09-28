# Phase 9 P9.0: control ownership, contract, and oracle freeze

Status: `PASS_P9_0_CONTROL_CONTRACT_AND_ORACLE_FREEZE`.

Baseline: `4688021624b017fd55e476466310e4908cd19b99` on
`next/pssolver-v0.2.0-architecture`.

P9.0 is a documentation and static-contract slice. It authorizes no
functional runtime implementation and changes no scientific or production
execution path. Its machine-readable authority is
[phase_9_p90_control_contract_freeze.json](phase_9_p90_control_contract_freeze.json),
and its architectural decision is [ADR 0013](adr/0013-functional-runtime-and-external-control-ownership.md).

## Prerequisite

Phase 8 is complete. Its final record qualifies four public model--geometry
applications and six runtime paths while preserving the legacy Plane and
Channel defaults. Phase 8 made Phase 9 planning eligible but did not implement
or authorize a differentiable runtime. P9.0 is the separately authorized
planning and freeze step.

## Ownership boundary

PSSolver owns:

- scientific equations and physical parameter meaning;
- geometry, boundary, transform, linear-solve, integrator, and execution
  identity;
- the model-declared physically admissible range of a controllable field;
- the precise equation term into which a field is injected;
- functional state layout and named observation conventions;
- pure functional execution and its differentiability declaration;
- deterministic-replay capability and replay identity;
- durable checkpoint import/export and production consistency validation.

An independent consumer owns:

- spatial actuator bases and temporal control parameterization;
- enforcement of optimization constraints within the physical admissibility
  range;
- objectives and scientific experiment diagnostics;
- checkpoint scheduling during an adjoint evaluation;
- forward/adjoint orchestration, optimizers, line searches, and campaigns.

The dependency is one way: a consumer may import PSSolver's public provisional
functional interface; PSSolver never imports the consumer. PSSolver documents
only the generic consumer contract, not a consumer repository, path, objective,
or experiment.

## Frozen legacy oracle

`pssolver.control` is a compatibility-public numerical oracle, not the new
architecture. Its current public names and behavior remain unchanged.

The source inventory records exact SHA-256 and Git-blob identities for:

- all eight `pssolver/control/*.py` files;
- the legacy Channel control driver;
- the production/functional forward-consistency script;
- the core Plane and Channel adjoint regression tests;
- the small three-dimensional DAL example.

The ownership decisions are:

| Legacy module | P9 ownership | Disposition |
|---|---|---|
| `active_force.py` | PSSolver model physics | later mechanical extraction to `models.active_nematics`, old path retained as a facade |
| `functional.py` | oracle only | replace with a public provisional functional runtime; do not extend |
| `controls.py` | independent consumer | keep compatibility surface, migrate research use outward |
| `dal.py` | independent consumer | keep compatibility surface, migrate research use outward |
| `objectives.py` | independent consumer | keep compatibility surface, migrate research use outward |
| `loop_diagnostics.py` | independent consumer | keep compatibility surface, migrate research use outward |
| `grid_transfer.py` | independent consumer utility | keep compatibility surface, migrate research use outward |
| `__init__.py` | compatibility facade | preserve names until the ADR 0005 deprecation window completes |

The one remaining production ownership debt is
`pssolver.channel -> pssolver.control.active_force`. P9.0 records it and does
not move it. No other production package module may add a dependency on
`pssolver.control`.

## Functional requirements R1--R12

The machine record freezes twelve requirements:

1. explicit provisional `FunctionalRuntime` construction and fail-closed
   capability declaration;
2. immutable flat-tuple state containing every persistent value that affects
   the next step;
3. pure step and combined step/observation semantics using the integrator's
   actual discrete map;
4. JSON scientific, discretization, and execution identity;
5. bitwise functional replay for one device, dtype, and execution identity;
6. named differentiable observations with input-state timing;
7. explicit field-control schema and equation injection;
8. model-owned physical admissibility bounds;
9. preflighted durable checkpoint conversion;
10. differentiability and inner-solve capability declarations;
11. optional explicit JVP/VJP and validated custom adjoints;
12. strict production/functional per-step consistency validation.

R1--R7, R10, and R12 block the first periodic integration. R8 and R9 are
required before production control campaigns. R11 is required when the
bounded-domain iterative pressure solve moves beyond its small-grid unrolled
reference.

## Frozen semantics

- `step_and_observe` returns the next state and the observation of the input
  state under the current control.
- That observation is bitwise identical to a separate observation call.
- A terminal observation without control omits control-dependent algebraic
  fields such as velocity.
- Functional state has an explicit batch axis. Batch size one is mandatory in
  the first slice; larger batches are an optional declared capability.
- Functional replay and checkpoint-stride gradient invariance are bitwise
  under a fixed execution identity.
- Production/functional comparison uses strict predeclared tolerances, not an
  unconditional byte-identity claim.
- Activity enters as `div(beta * alpha * Q)`, with the product formed before
  differentiation and its projection/dealiasing included in identity.
- Hidden warm starts and caches are forbidden unless they are state
  components. Diagnostic caches cannot alter future results.
- A concrete state packing is owned by each runtime identity; P9.0 does not
  freeze one permanent package-wide storage order.

## Slice order

1. **P9.0** freezes this contract and the legacy oracle.
2. **P9.1** qualifies periodic friction and z-invariant prerequisites and
   defines the provisional functional declarations.
3. **P9.2** implements the batch-one periodic complete-stress activity-control
   functional runtime.
4. **P9.3** adds deterministic replay and the functional/durable state bridge.
5. **P9.4** validates gradients, hidden-detach gates, and R12 consistency.
6. **P9.5** measures H100 forward/VJP/memory performance and optionally
   qualifies larger batches.
7. **P9.6** qualifies the periodic runtime through a first independent real
   consumer.
8. **P9.7** adds the Channel functional runtime, pressure differentiation,
   and a second independent consumer. Lifting and Robin capabilities qualify
   only when required by an actual combination.
9. **P9.8** stabilizes the public interface and performs installed-wheel H100
   closure.

The periodic consumer precedes Channel because periodic direct Fourier Stokes
has no pressure PCG, lifting, Robin wall, or bounded-domain zero-mode problem.
The interface remains provisional until both consumers pass.

## Non-claims and authorization

P9.0 does not provide a functional runtime, activity control field, gradient,
adjoint, optimizer, batch execution, public friction application, z-invariant
qualification, Channel pressure adjoint, H100 result, or control-science
result. It neither promotes a compiled runtime nor changes a production
default.

Local verification passed 74 focused tests and the complete suite of 2436
tests plus 8 subtests, with no failures, skips, deselections, or expected
failures. `git diff --check` passed. The future verbatim archive source list
now includes ADR 0013 and both P9.0 records, but the PDF was deliberately not
regenerated.

P9.0 is complete. P9.1 implementation, all H100 work, and all later Phase 9
slices remain separately unauthorized.
