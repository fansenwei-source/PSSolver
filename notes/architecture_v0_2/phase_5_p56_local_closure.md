# Phase 5 P5.6 local closure and Phase 6 handoff

Classification: `PASS_P5_6_LOCAL_CLOSURE`.

Baseline: P5.5 commit `e0388e6190b019ef438ab1a029970e4527e209ee`.

P5.6 closes the local Phase 5 contract.  It does not run the formal H100
qualification, change the production default, promote `compiled_v2`, connect
SBDF2 to Plane, or change the scientific model.

## Local closure result

The explicit `compiled_v2` path and `legacy_production` were compared with
fresh, identical float64 initial tensors at 1, 2, and 100 steps.  Physical
and spectral state were byte-identical at every endpoint.  Completed-step and
spectral-refresh clocks matched.  The compiled path retained the same owned
field, evolved-state, linear-operator, and denominator storage identities
through 100 steps; its timestep workspace remained a zero-slot, zero-byte
generation guard.

A 37+63 compiled checkpoint/restart matched a continuous 100-step compiled
run byte for byte.  A compiled checkpoint was rejected by a legacy target
before target state or progress changed.  Existing P5.4/P5.5 output,
diagnostic, metadata, COMPLETE, and application restart gates remain in the
complete CPU suite.

The compiled step AST was audited again.  It contains no registry or
configuration lookup, string field discovery, metadata construction,
unbounded loop, or direct tensor-allocation call.  Runtime fallback remains
forbidden.

## Installed-wheel workflow

A wheel was built from the working candidate, installed with `--no-deps` in
an isolated venv that reused the validated scientific dependencies, and
invoked outside the source checkout through the installed
`pssolver-plane-beris-edwards` entry point.  The explicit compiled runtime
completed two CPU steps, wrote Q/u/p and COMPLETE, and reported requested and
effective runtime `compiled_v2` with `fallback_used=false`.

The qualified wheel was `pssolver-0.1.2-py3-none-any.whl`, SHA-256
`10035feea3aa2402c476a7519bb2c4ec40998fd483a8eed236523380b0296cab`.

## Local CUDA result and device-identity correction

The first local CUDA construction exposed a connection-layer identity bug:
the transform backend retained the unindexed request `cuda`, whereas tensors
allocated by PyTorch correctly reported `cuda:0`.  The old strict
`torch.device` equality treated these names as different even though they
refer to the same selected accelerator.

The construction gate now accepts an omitted device index only when the
device type matches the concrete allocation.  An explicit index remains
strict.  This correction changes no equation, operation order, tensor layout,
or CPU result; it only lets the already-allocated GPU runtime pass its device
identity gate.  Unit tests cover CPU, unindexed CUDA, matching indexed CUDA,
wrong CUDA index, and wrong device type.

After the correction, an RTX 3060 Ti local CUDA smoke with PyTorch 2.6.0,
CUDA runtime 12.6, float64, and TF32 disabled passed at 1, 2, and 100 steps.
Legacy and compiled physical and spectral tensors were byte-identical at all
three endpoints.  No fallback, NaN, Inf, OOM, or CUDA error occurred.  This is
a local functional gate, not a substitute for the H100 performance gate.

The final Phase 5 targeted set passed 72 tests.  The complete local suite
passed 1,917 tests and 8 subtests with no failure, skip, deselection, or xfail.
`git diff --check` also passed.

## Phase boundary

Phase 5 is locally complete.  Phase 6 planning is eligible and its exact
contract is frozen in
`phase_6_plane_compiled_v2_qualification_plan.json`.  Phase 6 execution is
not authorized by this record.  `legacy_production` remains the implicit
default and rollback oracle; `compiled_v2` remains explicit opt-in.
