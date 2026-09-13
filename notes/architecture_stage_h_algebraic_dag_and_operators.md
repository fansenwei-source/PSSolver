# Architecture Stage H: algebraic DAGs, transient fields, and operators

## Scope and result

Stage H closes the structural gap between the Stage G body-force canary and a
real coupled constitutive model.  A physical model can now declare a frozen
pre-RHS dependency graph such as

```text
evolved Q -> transient H -> transient stress/force -> stored u,p -> Q RHS
```

without pretending that every intermediate tensor is an evolved field or a
checkpointed PDE unknown.  Mathematical gradient, Laplacian, and divergence
operations are exposed through a component-aware context; models do not name
FFT, DCT, DST, Hermitian packing, truncation, or the legacy `Fields` object.

This remains an opt-in architecture path.  It does not migrate the production
Beris--Edwards driver, alter a numerical default, or change the benchmark
`develop` worktree.

## H1: four distinct field lifetimes

`FieldRole` now distinguishes:

- `EVOLVED`: time-integrated physical unknowns with persistent storage;
- `ALGEBRAIC`: instantaneous physical unknowns with persistent runtime
  storage, such as velocity and pressure;
- `TRANSIENT`: instantaneous derived components cached only for the current
  synchronized pre-RHS state;
- `DIAGNOSTIC`: observations outside the PDE state.

Transient components receive complete boundary and spectral plans but no
storage index.  They are absent from legacy `Fields.spatial`,
`Fields.spectral`, static-field counts, and algebraic restart tensors.  The
legacy assembly records their deliberate omission separately from diagnostics.

The runtime metadata states that the transient cache is not legacy storage and
is not checkpointed.  A future complete checkpoint must reconstruct these
values from the restored evolved state by running algebraic synchronization.

## H2: frozen algebraic dependency DAG

`build_algebraic_execution_plan()` validates all systems before any numerical
solver is constructed.  It requires:

1. a unique system name;
2. exactly one owner for every algebraic or transient output component;
3. every dependency to be either evolved state or another declared output;
4. no self-dependency;
5. an acyclic system graph;
6. complete agreement between declared fields and owned outputs.

A stable topological sort freezes the execution order.  Declaration order and
execution order are both retained in metadata, together with component-labelled
dependency edges.  Independent ready systems retain declaration order, so the
schedule is deterministic.

Solver dispatch occurs in this frozen order.  Geometry selection is still the
exact `(geometry type, capability)` dispatch introduced in Stage F; dependency
ordering does not create a geometry fallback.

## H3: atomic transient cache

During one algebraic evaluation the adapter begins with evolved physical
components, executes systems in topological order, and makes a produced
component available to downstream systems only after validating its spectral
shape, dtype, device, and ownership.

Produced fields needed downstream are converted to their declared physical
component space.  Transient outputs are collected in a new local mapping and
published only after the entire DAG succeeds.  A partial or failed solve cannot
replace the previous cache.  The published mapping is read-only and carries a
monotonic generation counter.

The explicit RHS receives a read-only state mapping containing evolved,
stored-algebraic, and synchronized transient components.  Immediately after a
timestep, stored and transient algebraic values describe the pre-step evolved
state and are therefore stale together.  Explicit observation synchronization
refreshes both without advancing time.  Reset invalidates the transient cache,
clears implementation warm starts, rebuilds all algebraic outputs, and restores
the initial pre-RHS freshness flag.

The current legacy bridge invokes algebraic work through its static-field hook.
Consequently, a graph containing transient outputs but no stored algebraic
output is representable by the frozen plan but is rejected by this bridge.  A
future native runtime scheduler can remove that temporary compatibility limit.

## H4: mathematical operator context

`ModelExecutionContext` now includes the physical mathematical operations

```text
gradient(source component, output component, value, axis)
laplacian(component, value)
divergence(source components, output component, values)
```

Every operation checks tensor shape, dtype, device, axis, and the destination
component's boundary space.  For example, a wall-normal derivative maps a
Neumann component to a Dirichlet component; declaring the output as Neumann is
rejected instead of silently applying the wrong basis.

The context implements these operations with the qualified projected transform
backend, but that implementation detail is private.  `explicit_rhs` now
receives the same context explicitly, which prevents hidden global solver
access and makes the mathematical dependencies visible in the model contract.
Geometry-specific algebraic executors also receive this mathematical API plus
the restricted native spectral conversion required by a solver implementation.

## H5: chained qualification canary

`DifferentialAlgebraicChainModel` declares, deliberately in reverse order,

```text
screened_gradient:
    gradient_phi -> response

wall_normal_gradient:
    phi -> gradient_phi
```

where `phi` is evolved, `gradient_phi` is transient, and `response` is a stored
algebraic field.  Construction freezes the actual order as

```text
wall_normal_gradient -> screened_gradient
```

The explicit RHS consumes both `gradient_phi` and `response`.  The canary is
qualified with full-complex/full transforms and with the Plane-qualified
Hermitian-half/truncated path.

The Stage H tests cover:

- reverse declaration and deterministic topological execution;
- exact component-labelled dependency metadata;
- numerical reproduction of the gradient and Helmholtz chain;
- transient exclusion from legacy storage and restart state;
- cache immutability, staleness, synchronization, and reset generation;
- explicit RHS consumption of stored and transient values;
- direct model use of its mathematical operator context;
- analytic gradient, Laplacian, and divergence checks in mixed bases;
- destination-boundary mismatch rejection;
- missing dependency, duplicate owner, and cycle rejection;
- explicit rejection of the legacy transient-only compatibility case;
- continued isolation from production Plane, Channel, and benchmark paths.

Desktop CPU qualification after the final implementation:

- Stage B--H architecture plus existing Plane/Channel Stokes suites:
  `133 passed`;
- complete repository suite: `735 passed, 8 subtests passed`.

Both runs used an external Python bytecode cache and disabled the pytest cache.
No GPU qualification is required because Stage H does not select a production
path or change a numerical default.

## Production boundary after Stage H

Stage H supplies the lifecycle and dependency machinery needed by the real
Beris--Edwards chain, but it does not yet implement that physics.  The next
stage should migrate and equation-test the constitutive pieces in dependency
order: molecular field, passive/active stress, complete stress divergence, and
only then the existing geometry-specific Stokes capability.  Production driver
migration, long-trajectory equivalence, GPU performance qualification, default
promotion, and Channel-specific migration remain later gates.
