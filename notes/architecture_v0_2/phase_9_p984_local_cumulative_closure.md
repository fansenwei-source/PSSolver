# P9.8.4 local cumulative CPU and installed-wheel closure

Status: `PASS_P9_8_4_LOCAL_CUMULATIVE_CPU_INSTALLED_WHEEL_CLOSURE`

P9.8.4 closes the local part of the stable functional API migration. Clean
archives of PSSolver commit `710df30` and PSSolver-Control commit `ea24027`
were used to build wheels. Both wheels were installed together outside the
source checkouts, and the Periodic and Channel consumers were exercised only
through `pssolver.functional.api` version `1.0`.

## Bound artifacts

| Artifact | SHA-256 |
|---|---|
| PSSolver source archive | `790cef37b6305c9e51fd03d15b35cb0d0847dbe2a37dfba7390f2dd1710eb641` |
| PSSolver-Control source archive | `b50f86e4a9db6ee37f24435f5773636d32e167539374a7e5dbc595dc9e092827` |
| `pssolver-0.1.2-py3-none-any.whl` | `5159aeb4939f2e689d1ca1ec02139491edcc6d963fc0106bfb75e5ebe95c2388` |
| `pssolver_control-0.0.1.dev0-py3-none-any.whl` | `54ae5af30336e85da5495bc0b5638375b790b67cf883f56b93e8639c4aa1567c` |

The installed imports for both packages, the stable functional API, and both
consumer adapters resolve from the qualification environment's
`site-packages`; neither source checkout appears on the runtime import path.
The installed PSSolver console entry point also imports successfully.

## Cumulative local gates

The immediately preceding clean source revisions retain their complete-suite
results:

- PSSolver: `2607 passed, 8 subtests passed`;
- PSSolver-Control: `235 passed, 1 skipped`, with only the opt-in R5 benchmark
  skipped.

The external installed-wheel harness reports `92 passed`. It covers stable API
identity, both independent consumers, current and provisional checkpoint
readers, replay, restart and production continuation, checkpointed and
full-history gradients, finite differences, Taylor behavior, negative gates,
and import boundaries. Source copies used for static AST audits and fixtures
were not placed on `sys.path`.

The declared requirements of PSSolver and PSSolver-Control are satisfied. A
global `pip check` exposes a pre-existing, unrelated metadata conflict between
the desktop installation of `nematics3d 0.9.0b1` and NumPy 2.4.3. Neither
qualified wheel depends on `nematics3d`, and the shared environment was not
modified. The record therefore reports the global conflict explicitly and
uses a per-distribution requirement audit; it does not claim that the global
environment passed `pip check`.

## Boundary

P9.8.4 authorizes planning for P9.8.5 only. It does not authorize or execute an
H100 job, change production defaults, qualify larger batches, authorize an
optimizer campaign, make a scientific-control claim, or complete Phase 9.
P9.8.5 remains the single-H100 cumulative closure. The cumulative verbatim PDF
is not regenerated in this step.
