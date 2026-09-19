# ADR 0004: geometry-specific capability dispatch

Status: accepted.

## Context

Plane and Channel share spectral primitives and active-nematic constitutive
physics, but their pressure, parity, wall, zero-mode, and Schur-complement
structures are not interchangeable.  Treating geometry-specific solvers as an
implementation smell would either hide physical assumptions or force all
geometries through a slower generic path.

## Decision

Algebraic solvers are selected by an exact capability key containing geometry
and the relevant boundary/discretization signature.  Unsupported combinations
fail closed.  There is no implicit fallback from Channel to Plane, from mixed
BCs to periodic BCs, or from one pressure/zero-mode convention to another.

Shared layers provide transform primitives, generic differential operators,
layout contracts, and solver protocols.  Dedicated implementations provide
periodic, Plane, and Channel Poisson/Stokes/Schur algorithms.

## Consequences

- Plane qualification does not promote Channel.
- Each geometry requires its own manufactured tests, boundary residuals,
  pressure/gauge checks, memory measurements, and performance qualification.
- Fast geometry-specific algorithms remain first-class implementations under
  a common contract.
- A generic reference implementation may exist for verification without
  becoming the production default.

## Boundary extension

Strong anchoring and nonhomogeneous wall laws are bound to geometric faces but
do not live in the geometry module.  Planning may choose lifting, a compatible
eigenbasis, tau/bordering, or a surface-energy closure according to the
physical boundary assignment and solver capability.

