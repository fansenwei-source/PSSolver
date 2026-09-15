# Architecture Stage O.4.2: tensor-residency diagnosis

Stage O.4.1 established that the separated Plane canary is scientifically
equivalent and structurally stable, but its R320 timestep and CUDA-memory
footprint regress with grid volume. Stage O.4.2 is therefore a measurement
stage, not another promotion attempt.

The diagnostic inventories CUDA tensors reachable from an explicitly named
legacy solver or separated runtime root. It records tensor shape, stride,
dtype, view status, owner path, and category, while counting shared backing
storage once. It does not use the global garbage collector and does not retain
tensor objects in its JSON result. CUDA allocator counters are captured beside
the inventory so the report distinguishes runtime-owned storage from allocator
bytes that the bounded traversal cannot attribute. Overlapping device-address
ranges are coalesced before unique bytes are reported. Any remaining
inventory/allocator inconsistency is persisted as diagnostic evidence; it is
not allowed to suppress the profile JSON.

Selected `cat`, `stack`, `clone`, `contiguous`, `copy`, allocation, and device
conversion operators are profiled in a separate window. Their memory fields
describe allocator effects, not complete read/write traffic. Semantic-region
timings are also retained, but Stage O.4.1 remains the authoritative balanced
performance evidence. Two unmeasured settling steps separate observation from
the operator audit so observation-specific integrator state is not counted as
a normal timestep.

When a runtime-root inventory and `torch.cuda.memory_allocated()` disagree,
each reported storage address range is also reconciled against the active,
awaiting-free, and inactive blocks in `torch.cuda.memory_snapshot()`. The
report records exact unmatched owner paths without retaining tensors or the
allocator snapshot. Allocator counters are sampled before the inventory,
immediately before the block snapshot, and after that snapshot so a changing
measurement window cannot be mistaken for an ownership discrepancy. This is
an accounting diagnostic only; unmatched or unstable evidence cannot
authorize an optimization.

The storage-identity query itself may materialize allocator bookkeeping for
already reachable transient tensors. Each phase therefore performs one
discarded, unmeasured priming inventory before its formal baseline. The second
inventory is compared with the priming inventory at three separate levels:
unique storage identity, storage-to-owner paths, and complete tensor-reference
records. A third inventory must then be exactly identical to the second one,
which independently verifies the post-priming state. All three inventories,
canonical hashes, aggregate-field differences, and bounded path-level
additions/removals are preserved. Exact equality remains a reported fact, but
it is not conflated with storage identity: a reference-only expansion can be
classified separately from a newly allocated or removed storage. Only exact
equality can close the accounting gate; any reference-graph change or any
second-to-third-pass change keeps it closed. In particular, only built-in
dictionaries and read-only mapping proxies are traversed through key lookup.
Custom `Mapping` implementations are inspected through their stored attributes
so a supposedly read-only inventory cannot trigger a lazy `__getitem__` and
materialize application state. All post-priming allocator samples must also
remain stable. Priming deltas are preserved as evidence instead of being
folded into runtime-residency comparisons.

The H100 plan runs exactly one R320 diagnostic for each runtime and one
read-only comparison. A successful result may authorize a narrowly targeted
Stage O.4.3 optimization design. It cannot change the runtime default, promote
the canary, or alter Plane/Channel equations.
