# Architecture Stage O.4.3.1: natural-storage transform views

## Decision from Stage O.4.3

The packed-publication candidate is rejected and must remain opt-in. Its
100-step trajectory was bitwise identical to the baseline and it reduced
copy-cat batches from 180 to 80, but the R320 timestep was 3.016% slower,
peak allocated memory was 26.30% higher, and peak reserved memory was 28.15%
higher. The result is a real implementation failure, not timing noise or a
scientific-tolerance issue.

The cause is architectural. A generation-wide packed allocation is filled by
copying every independently produced algebraic spectrum. It therefore adds
full-field writes and keeps a second large generation store alive merely to
make later transform inputs adjacent. Avoiding selected `torch.cat` calls
does not repay that publication and lifetime cost.

## Bounded replacement candidate

Stage O.4.3.1 changes one variable only:

- baseline: `deferred_stack` plus `copy_cat`;
- candidate: `deferred_stack` plus `contiguous_storage_view`.

The candidate does not republish algebraic outputs and does not allocate a
generation-wide transient store. It opportunistically reuses only tensors
that a producer already returned as adjacent views of one allocation. Every
distinct, non-contiguous, non-adjacent, or otherwise unsafe group falls back
to the historical `torch.cat` path.

This is intentionally smaller than the rejected candidate. A coupled CPU
canary already proves that naturally adjacent groups exist, that copy-cat
counts decrease, and that complete spatial and spectral timesteps remain
bitwise identical. Only H100 measurements can determine whether the saved
copies exceed the conservative storage-validation overhead at R320.

## Qualification boundary

The H100 test binds both the closed Stage O.4.2 accounting evidence and the
Stage O.4.3 rejection report. It repeats the 100-step Q/u/p comparison and
three balanced R128/R320 profile trials. It retains the existing numerical,
performance, paired-timing, memory, and safety gates.

An `A_recommended` result authorizes only a later Stage O.4.4 architecture
decision. A `B_neutral` or `C_rejected` result closes opportunistic runtime
storage inspection as a performance path; the next design must move packed
ownership to the producing operator instead of copying after production.

No Stage O.4.3.1 code is imported by production Plane, Channel, or the generic
solver, and no production default changes in this stage.
