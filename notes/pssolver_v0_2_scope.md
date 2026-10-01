# PSSolver v0.2 support scope

PSSolver v0.2 is a research-grade tensor-product spectral solver with a typed,
composable public declaration layer.  The v0.2 support claim is bounded by the
capability catalog, installed-wheel qualification evidence, and the explicit
functional protocol described below.  It is not a claim that arbitrary models,
geometries, boundary laws, or controls are executable.

## Public composition boundary

The package-root `Simulation` declaration assigns separate ownership to the
model, geometry, field-level boundary policies, spectral numerics, time
integration, initial condition, execution policy, output workflow,
discretization controls, and invocation provenance.  `compile_simulation`
resolves one registered application before allocation; `run_simulation`
executes only that compiled route.  There is no runtime fallback or per-step
registry dispatch.

The public capability catalog qualifies four executable combinations:

1. complete-stress Beris--Edwards on a Plane slab;
2. legacy active-force active nematics in a rectangular Channel;
3. complete-stress Beris--Edwards in a periodic box;
4. complete-stress Beris--Edwards in a rectangular Channel.

Plane and legacy Channel retain their established production defaults.
Compiled variants remain explicit opt-ins and retain rollback identities.

## Boundary capability

Homogeneous Q Neumann, free-slip velocity, no-slip velocity, and pressure
compatibility policies are typed field-level declarations.  The Plane
complete-stress application additionally qualifies static prescribed Q data
and strong homeotropic/planar Q policies through the static lifting route.

Finite-Q Robin relaxation passed internal numerical, restart, conditioning,
package, and H100 gates, but remains an internal pilot.  General public Robin
execution, nonhomogeneous Neumann lifting, dynamic boundary data, trainable
boundary data, and arbitrary-field lifting are not claimed.

## Stable functional boundary

`pssolver.functional.api` protocol 1.0 is the supported independent-consumer
surface.  It qualifies batch-one Periodic and rectangular-Channel activity
runtimes with deterministic state transitions, observations, diagnostics,
checkpoint bridges, replay, and differentiable execution.  The two checkpoint
formats are versioned separately and reject incompatible identities or payload
integrity failures before target mutation.

The Periodic runtime projects the half-spectrum self-conjugate planes back to
the Hermitian subspace every step.  The Channel functional runtime uses a pure
zero-start pressure solve and its custom implicit pressure adjoint; the
production Channel path retains its pressure warm start.

## Compatibility

The historical Plane APIs, CLI, compatibility script, transform facade, and
legacy runtime remain available.  Functional compatibility aliases follow the
deprecation window frozen in Phase 9: deprecated in package 0.2.0 and not
removable before 0.4.0.  Silent fallback and heuristic checkpoint repair are
forbidden.

## Explicit exclusions

PSSolver v0.2 does not claim:

- arbitrary PDE or arbitrary model--geometry execution;
- inertial or paper-identical Shendruk reproduction;
- paper-identical initialization;
- functional batches larger than one;
- Plane optimal-control qualification;
- production-scale optimizer or memory qualification;
- an authorized optimizer campaign or scientific control result;
- public nonhomogeneous Neumann, dynamic/trainable, or general Robin boundary
  execution;
- modification, bundling, or qualification of `nematics3d`.

PSSolver-Control is an independent consumer of the stable functional API and
is versioned and distributed separately.
