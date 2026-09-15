# Architecture Stage O.4.3.2: producer-owned packing contract

## Why O.4.3.1 closes without promotion

The natural-storage-view candidate is scientifically exact and memory neutral,
but it is performance neutral. At R320 its candidate/baseline timestep ratio
was `1.000066`; peak allocated and reserved memory ratios were both `1.0`.
Only one multi-component batch per timestep was naturally adjacent, reducing
copy-cat calls from nine to eight. The result is correctly `B_neutral`.

This closes repeated runtime storage inspection as an optimization path. The
scheduler cannot manufacture adjacency for independently allocated producer
outputs without either copying them, as rejected in O.4.3, or changing the
owner of their allocation.

## Observed producer boundary

A coupled CPU trace reproduces the H100 count and assigns the remaining
multi-component copies to these stable groups:

1. five Q explicit-RHS components projected forward;
2. five molecular-field components projected forward;
3. fourteen wall-even stress components projected forward;
4. four wall-odd stress components projected forward;
5. fifteen wall-even H/Q-gradient components transformed inverse;
6. five wall-odd Q-gradient components transformed inverse;
7. five wall-even velocity-gradient components transformed inverse;
8. four wall-odd velocity-gradient components transformed inverse.

The force-x/force-y forward projection is already the single natural zero-copy
batch. Force-z is a singleton.

## New ownership contract

`ProducerPackedComponentValues` is a narrow, experimental mapping over one
producer-owned tensor with shape `(component, batch, *grid)`. Construction:

- never stacks, concatenates, copies, or reorders data;
- exposes named component views over the exact producer allocation;
- can expose a flattened transform view only for a contiguous component run;
- rejects invalid names, shapes, layouts, empty storage, and non-contiguous
  storage;
- reports tensor-free ownership metadata;
- retains no tensor other than the producer allocation it represents.

The existing conservative transform scheduler consumes its component views
without a new special case. Therefore storage validation, boundary grouping,
and mandatory copy-cat fallback remain centralized in the scheduler.

## Stage boundary

O.4.3.2 is CPU architecture groundwork, not an O.4.4 qualification and not a
production promotion. It does not yet change any Beris--Edwards producer. The
next candidate should adapt one bounded producer family so its native kernel
result has the required component order. For pointwise compiled producers,
the packed output must be written directly by the fused producer kernel; a
post-kernel `stack`, `cat`, or generation-wide copy is not acceptable.

Only after CPU equivalence and ownership tests pass should that candidate get
an H100 A/B gate. Production Plane, Channel, the generic solver, equations,
transform order, and all defaults remain unchanged.
