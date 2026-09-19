# ADR 0005: public API and compatibility policy

Status: accepted.

## Context

The package root exports historical construction classes and transform
implementations, while the supported v0.1 application API lives in package
submodules.  New declarative modules are already importable but have not yet
earned a stable v1 contract.  Mechanical module extraction must not break
existing research scripts.

## Decision

API names are classified as:

1. **supported**: qualified application entry points and their documented
   input/output contracts;
2. **compatibility-public**: historical imports that remain available through
   facades but are not preferred for new architecture code;
3. **provisional**: typed architecture contracts available for internal and
   early-adopter use but not yet stable through v1;
4. **internal/experimental**: implementation details, qualification tools,
   and unsupported paths.

Moving an implementation does not remove its old import path.  Compatibility
facades preserve names and call signatures.  Removing a compatibility-public
name requires an announced deprecation spanning at least two minor releases
and a migration note.

`__all__` records an import surface; before v1 it does not by itself promise a
permanent ABI or implementation module.

## Production boundary

Production paths may not acquire new imports from `pssolver.experimental`.
The single v0.1.2 runtime bridge to the separated canary is an exact,
documented migration debt and cannot be generalized into a package-wide
exception.

## Consequences

Phase 1 can split `transforms.py` without forcing downstream users to migrate
immediately.  New code uses canonical modules, while old code continues to
import from `pssolver.transforms`.
