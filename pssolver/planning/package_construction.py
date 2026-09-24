"""Tensor-free ownership-normalized runtime construction plans.

P7.7.5 layers these products over the P7.7.4 runtime binding.  A plan records
which package implementation owns construction, but contains no callable,
tensor, runtime object, registry, or mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .construction import RuntimeConstructionKind


PACKAGE_RUNTIME_CONSTRUCTION_PLAN_SCHEMA_VERSION = 1


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


@dataclass(frozen=True, slots=True)
class PackageRuntimeConstructionPlan:
    """Immutable package-ownership decision for one qualified binding."""

    source_binding_sha256: str
    source_simulation_sha256: str
    lowering_plan_sha256: str
    kind: RuntimeConstructionKind
    request_type: str
    package_factory: str
    implementation_builder: str
    adapter_protocol: str
    fallback_allowed: bool = False

    def __post_init__(self) -> None:
        _sha256(self.source_binding_sha256, "source binding identity")
        _sha256(self.source_simulation_sha256, "source simulation identity")
        _sha256(self.lowering_plan_sha256, "lowering plan identity")
        if not isinstance(self.kind, RuntimeConstructionKind):
            raise TypeError("kind must be a RuntimeConstructionKind")
        for value, description in (
            (self.request_type, "request type"),
            (self.package_factory, "package factory"),
            (self.implementation_builder, "implementation builder"),
            (self.adapter_protocol, "adapter protocol"),
        ):
            _qualified_name(value, description)
        if self.fallback_allowed is not False:
            raise ValueError("package runtime construction forbids fallback")

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": PACKAGE_RUNTIME_CONSTRUCTION_PLAN_SCHEMA_VERSION,
            "source_binding_sha256": self.source_binding_sha256,
            "source_simulation_sha256": self.source_simulation_sha256,
            "lowering_plan_sha256": self.lowering_plan_sha256,
            "kind": self.kind.value,
            "request_type": self.request_type,
            "package_factory": self.package_factory,
            "implementation_builder": self.implementation_builder,
            "builder_provision": "package",
            "adapter_protocol": self.adapter_protocol,
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
    "PACKAGE_RUNTIME_CONSTRUCTION_PLAN_SCHEMA_VERSION",
    "PackageRuntimeConstructionPlan",
]
