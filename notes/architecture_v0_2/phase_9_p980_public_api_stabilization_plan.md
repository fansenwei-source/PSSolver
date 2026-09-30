# Phase 9 P9.8.0: public functional API stabilization plan

## Objective

P9.8 converts the functional interface validated by the independent periodic
and Channel consumers from a provisional research contract into a documented,
versioned, installed-package public contract.  It then closes Phase 9 with one
cumulative installed-wheel CPU and H100 qualification.

P9.8 is an interface-stabilization phase.  It is not an opportunity to change
the Beris--Edwards equations, spectral discretization, boundary treatment,
time integrator, pressure operator, production defaults, or qualified runtime
identities.

## Frozen prerequisites

The two independent consumer closures are prerequisites:

- P9.6: periodic complete-stress batch-one activity consumer;
- P9.7.6: rectangular-Channel complete-stress batch-one activity consumer.

Both use only installed public PSSolver interfaces.  Neither result authorizes
larger batches, a new control variable, an optimizer campaign, or a scientific
control claim.

## Stable surface to inventory

P9.8.1 first records, without changing behavior, every public functional name,
constructor signature, immutable declaration, state and control layout,
runtime property, observation contract, checkpoint field, identity digest,
exception type, and package export used by either qualified consumer.  The
inventory must distinguish:

1. stable public provider contracts;
2. compatibility-public facades retained under ADR 0005;
3. consumer-owned adapter, objective, checkpoint schedule, and optimizer
   contracts;
4. internal implementation details that remain unsupported.

No private member used only by a qualification harness becomes public merely
because it appeared in a test.

## Versioning and compatibility

P9.8.2 assigns a stable functional protocol version only after the inventory
is reviewed.  P9.8.0 does not preselect that literal version and does not alter
the current `0.1-provisional` value.  The implementation must provide:

- an explicit supported-version negotiation rule;
- a deterministic refusal for unknown versions and incompatible runtime or
  checkpoint identities;
- compatibility readers or a documented migration for every checkpoint
  schema written by the qualified P9.6 and P9.7.6 paths;
- at least the two-minor-release deprecation window required by ADR 0005 for
  compatibility-public names;
- machine-readable package and protocol version provenance.

Silent fallback, heuristic schema repair, source-worktree shadowing, and
private-provider imports remain forbidden.

## Pressure convergence metadata

P9.8 must remove the ambiguity exposed by the P9.7.6 evidence.  For Channel
pressure and transpose-pressure solves, stable metadata must report at least:

- solver and operator identity;
- requested maximum or fixed iteration count;
- achieved iteration count;
- achieved absolute and relative residual;
- requested tolerance;
- whether the tolerance is an early-convergence criterion or a required
  postcondition;
- the termination reason;
- whether the result is acceptable under the selected convergence mode.

The existing fixed-iteration numerical path remains unchanged unless a later,
separately qualified change is authorized.  P9.8 stabilizes its meaning; it
does not rewrite PCG.

## Planned slices

1. **P9.8.1 -- surface and schema inventory.**  Record exact provider and
   consumer usage, checkpoint versions, identity fields, exceptions, and
   package exports.  No runtime change.
2. **P9.8.2 -- stable protocol and compatibility facade.**  Freeze names,
   version negotiation, compatibility readers, refusal semantics, and
   pressure-convergence metadata.
3. **P9.8.3 -- packaging, documentation, and consumer migration.**  Update
   both installed-wheel consumers to the stable protocol without private
   imports or source shadowing.  Preserve old qualified checkpoints through
   the declared compatibility path.
4. **P9.8.4 -- local cumulative CPU closure.**  Run provider and consumer
   suites, installed-wheel tests, cross-version checkpoint tests, replay,
   restart, gradient, negative, and import-boundary gates.
5. **P9.8.5 -- single-H100 cumulative closure.**  Requalify the periodic and
   Channel batch-one consumers, functional/production agreement, gradients,
   memory, convergence metadata, checkpoint migration, and negative gates.
6. **P9.8.6 -- Phase 9 final record.**  Record the exact public surface and
   evidence, update the architecture archive source list, and decide whether
   the package is eligible for a versioned release.  PDF regeneration remains
   a separate explicit user action.

## Acceptance boundary

Phase 9 is complete only if both qualified consumers work from installed
wheels under the stable protocol; old qualified checkpoints have the declared
compatibility behavior; all identities and convergence outcomes are
machine-readable; CPU and the single H100 closure pass; and the final record
contains no unsupported scientific claim.

P9.8.0 authorizes planning of P9.8.1 only.  It does not implement a stable
version, modify runtime code, authorize H100 work, promote a production
default, qualify batch sizes above one, or authorize optimization and science
campaigns.
