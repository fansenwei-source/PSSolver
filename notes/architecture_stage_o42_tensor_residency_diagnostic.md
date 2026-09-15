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

The H100 plan runs exactly one R320 diagnostic for each runtime and one
read-only comparison. A successful result may authorize a narrowly targeted
Stage O.4.3 optimization design. It cannot change the runtime default, promote
the canary, or alter Plane/Channel equations.
