"""Tensor-free construction bindings for qualified simulation runtimes.

The products in this module identify an already-qualified construction edge.
They deliberately contain no tensors, callables, registries, solver objects,
or runtime state.  A runtime composition root may consume a binding once;
the binding must never be consulted from a timestep hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json


RUNTIME_CONSTRUCTION_BINDING_SCHEMA_VERSION = 1


def _qualified_name(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    if any(not part.isidentifier() for part in value.split(".")):
        raise ValueError(f"{description} must be a dotted Python identifier")
    return value


def _sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


class RuntimeConstructionKind(str, Enum):
    """Closed set of runtime targets qualified before P7.7.4."""

    PLANE_LEGACY_PRODUCTION = "plane_legacy_production"
    PLANE_COMPILED_V2 = "plane_compiled_v2"
    CHANNEL_LEGACY = "channel_legacy"
    CHANNEL_COMPILED_V2 = "channel_compiled_v2"


class BuilderProvision(str, Enum):
    """Owner responsible for supplying the selected numerical builder."""

    PACKAGE = "package"
    CALLER = "caller"


@dataclass(frozen=True, slots=True)
class RuntimeConstructionBinding:
    """Immutable identity of one prevalidated runtime-construction edge."""

    source_simulation_sha256: str
    lowering_plan_sha256: str
    equation_variant: str
    geometry_name: str
    runtime_path: str
    kind: RuntimeConstructionKind
    request_type: str
    runtime_factory: str
    adapter_protocol: str
    solver_implementation: str
    builder_provision: BuilderProvision
    builder_parameter: str | None
    fallback_allowed: bool = False

    def __post_init__(self) -> None:
        _sha256(self.source_simulation_sha256, "source simulation identity")
        _sha256(self.lowering_plan_sha256, "lowering plan identity")
        for value, description in (
            (self.equation_variant, "equation variant"),
            (self.geometry_name, "geometry name"),
            (self.runtime_path, "runtime path"),
        ):
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{description} must be a Python identifier")
        if not isinstance(self.kind, RuntimeConstructionKind):
            raise TypeError("kind must be a RuntimeConstructionKind")
        for value, description in (
            (self.request_type, "request type"),
            (self.runtime_factory, "runtime factory"),
            (self.adapter_protocol, "adapter protocol"),
            (self.solver_implementation, "solver implementation"),
        ):
            _qualified_name(value, description)
        if not isinstance(self.builder_provision, BuilderProvision):
            raise TypeError("builder_provision must be a BuilderProvision")
        parameter = self.builder_parameter
        if parameter is not None and (
            not isinstance(parameter, str) or not parameter.isidentifier()
        ):
            raise ValueError(
                "builder_parameter must be None or a Python identifier"
            )
        if self.builder_provision is BuilderProvision.CALLER and parameter is None:
            raise ValueError("caller-provided bindings require a builder parameter")
        if self.builder_provision is BuilderProvision.PACKAGE and parameter is not None:
            raise ValueError(
                "package-provided bindings cannot name a builder parameter"
            )
        if self.fallback_allowed is not False:
            raise ValueError("qualified runtime construction forbids fallback")

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_CONSTRUCTION_BINDING_SCHEMA_VERSION,
            "source_simulation_sha256": self.source_simulation_sha256,
            "lowering_plan_sha256": self.lowering_plan_sha256,
            "equation_variant": self.equation_variant,
            "geometry_name": self.geometry_name,
            "runtime_path": self.runtime_path,
            "kind": self.kind.value,
            "request_type": self.request_type,
            "runtime_factory": self.runtime_factory,
            "adapter_protocol": self.adapter_protocol,
            "solver_implementation": self.solver_implementation,
            "builder_provision": self.builder_provision.value,
            "builder_parameter": self.builder_parameter,
            "fallback_allowed": self.fallback_allowed,
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "BuilderProvision",
    "RUNTIME_CONSTRUCTION_BINDING_SCHEMA_VERSION",
    "RuntimeConstructionBinding",
    "RuntimeConstructionKind",
]
