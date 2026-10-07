"""Typed, tensor-free identities for future checkpoint compatibility.

RC4.2.1 deliberately does not connect these primitives to any checkpoint
reader or writer.  In particular, importing this module cannot broaden or
narrow the checkpoint formats accepted by the Plane, Periodic, Channel, or
functional workflows.  It freezes the vocabulary and legacy-schema catalog
that later RC4.2 slices must use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType


IDENTITY_SCHEMA_VERSION = 1
LEGACY_REGISTRY_SCHEMA_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]*$")


class IdentityLayerKind(str, Enum):
    """The six identity layers frozen by RC4.2.0."""

    RUN_PROVENANCE = "run_provenance"
    FORWARD_DYNAMICS = "forward_dynamics"
    DERIVATIVE_DYNAMICS = "derivative_dynamics"
    STATE_LAYOUT = "state_layout"
    BACKEND_RESTART = "backend_restart"
    MATERIALIZATION_PROVENANCE = "materialization_provenance"


class CheckpointFamily(str, Enum):
    """Durable checkpoint families with independently versioned schemas."""

    PLANE_WORKFLOW = "plane_workflow"
    PERIODIC_WORKFLOW = "periodic_workflow"
    CHANNEL_WORKFLOW = "channel_workflow"
    PERIODIC_FUNCTIONAL = "periodic_functional"
    CHANNEL_FUNCTIONAL = "channel_functional"


class SourceReleaseGeneration(str, Enum):
    """Released generations whose legacy schemas are frozen here."""

    RC1 = "rc1"
    RC2 = "rc2"
    RC3 = "rc3"


class LegacyDisposition(str, Enum):
    """What registration alone permits a future reader to claim."""

    ADJUDICATION_REQUIRED = "adjudication_required"
    STATE_MIGRATION_ONLY = "state_migration_only"
    REJECT_EXACT_RESTART = "reject_exact_restart"


_COMPATIBILITY_LAYERS = frozenset(
    {
        IdentityLayerKind.FORWARD_DYNAMICS,
        IdentityLayerKind.DERIVATIVE_DYNAMICS,
        IdentityLayerKind.STATE_LAYOUT,
        IdentityLayerKind.BACKEND_RESTART,
    }
)


def _require_positive_version(value: object, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{description} must be a positive integer")
    return value


def _require_token(value: object, description: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase identity token")
    return value


def _json_value(value: object, description: str) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} keys must be strings")
            result[key] = _json_value(item, description)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item, description) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} must not contain NaN or Inf")
        return value
    raise TypeError(f"{description} must be JSON-compatible")


def _canonical_json(value: object, description: str) -> str:
    normalized = _json_value(value, description)
    return json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointIdentityLayer:
    """One immutable and canonically hashed identity layer."""

    kind: IdentityLayerKind
    version: str
    payload: Mapping[str, object]
    schema_version: int = IDENTITY_SCHEMA_VERSION
    _canonical: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, IdentityLayerKind):
            raise TypeError("kind must be an IdentityLayerKind")
        _require_positive_version(self.schema_version, "schema_version")
        _require_token(self.version, "identity layer version")
        if not isinstance(self.payload, Mapping):
            raise TypeError("identity layer payload must be a mapping")
        metadata = {
            "kind": self.kind.value,
            "payload": _json_value(self.payload, "identity layer payload"),
            "schema_version": self.schema_version,
            "version": self.version,
        }
        canonical = _canonical_json(metadata, "identity layer")
        object.__setattr__(
            self,
            "payload",
            _freeze_json(json.loads(canonical)["payload"]),
        )
        object.__setattr__(self, "_canonical", canonical)

    @property
    def checkpoint_compatibility_gate(self) -> bool:
        return self.kind in _COMPATIBILITY_LAYERS

    def canonical_sha256(self) -> str:
        return _sha256_text(self._canonical)

    def __hash__(self) -> int:
        return hash(self._canonical)

    def to_metadata(self) -> dict[str, object]:
        return json.loads(self._canonical)


@dataclass(frozen=True, slots=True)
class CheckpointCompatibilityIdentity:
    """The typed identity used by a future exact-restart decision.

    Run and materialization provenance are intentionally absent.  A caller
    cannot accidentally make requested device or fresh-construction details
    part of this digest by attaching those layers to the aggregate.
    """

    family: CheckpointFamily
    runtime_path: str
    forward_dynamics: CheckpointIdentityLayer
    state_layout: CheckpointIdentityLayer
    backend_restart: CheckpointIdentityLayer
    derivative_dynamics: CheckpointIdentityLayer | None = None
    schema_version: int = IDENTITY_SCHEMA_VERSION
    _canonical: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.family, CheckpointFamily):
            raise TypeError("family must be a CheckpointFamily")
        _require_token(self.runtime_path, "runtime_path")
        _require_positive_version(self.schema_version, "schema_version")
        required = (
            (self.forward_dynamics, IdentityLayerKind.FORWARD_DYNAMICS),
            (self.state_layout, IdentityLayerKind.STATE_LAYOUT),
            (self.backend_restart, IdentityLayerKind.BACKEND_RESTART),
        )
        for layer, expected in required:
            if not isinstance(layer, CheckpointIdentityLayer):
                raise TypeError(f"{expected.value} must be an identity layer")
            if layer.kind is not expected:
                raise ValueError(
                    f"expected {expected.value}, got {layer.kind.value}"
                )
        derivative = self.derivative_dynamics
        if derivative is not None:
            if not isinstance(derivative, CheckpointIdentityLayer):
                raise TypeError(
                    "derivative_dynamics must be an identity layer or None"
                )
            if derivative.kind is not IdentityLayerKind.DERIVATIVE_DYNAMICS:
                raise ValueError("derivative_dynamics has the wrong layer kind")
        if self.family in {
            CheckpointFamily.PERIODIC_FUNCTIONAL,
            CheckpointFamily.CHANNEL_FUNCTIONAL,
        }:
            if derivative is None:
                raise ValueError(
                    "functional checkpoint identities require derivative dynamics"
                )
        elif derivative is not None:
            raise ValueError(
                "production checkpoint identities must not carry derivative dynamics"
            )
        metadata = {
            "backend_restart": self.backend_restart.to_metadata(),
            "derivative_dynamics": (
                None if derivative is None else derivative.to_metadata()
            ),
            "family": self.family.value,
            "forward_dynamics": self.forward_dynamics.to_metadata(),
            "runtime_path": self.runtime_path,
            "schema_version": self.schema_version,
            "state_layout": self.state_layout.to_metadata(),
        }
        object.__setattr__(
            self,
            "_canonical",
            _canonical_json(metadata, "checkpoint compatibility identity"),
        )

    def canonical_sha256(self) -> str:
        return _sha256_text(self._canonical)

    def __hash__(self) -> int:
        return hash(self._canonical)

    def to_metadata(self) -> dict[str, object]:
        return json.loads(self._canonical)


@dataclass(frozen=True, slots=True)
class LegacyIdentityKey:
    """Exact key for a frozen legacy identity schema."""

    checkpoint_family: CheckpointFamily
    format_version: int
    source_release_generation: SourceReleaseGeneration
    runtime_path: str
    applicability_class: str

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_family, CheckpointFamily):
            raise TypeError("checkpoint_family must be a CheckpointFamily")
        _require_positive_version(self.format_version, "format_version")
        if not isinstance(
            self.source_release_generation,
            SourceReleaseGeneration,
        ):
            raise TypeError(
                "source_release_generation must be a SourceReleaseGeneration"
            )
        _require_token(self.runtime_path, "runtime_path")
        _require_token(self.applicability_class, "applicability_class")

    def to_metadata(self) -> dict[str, object]:
        return {
            "applicability_class": self.applicability_class,
            "checkpoint_family": self.checkpoint_family.value,
            "format_version": self.format_version,
            "runtime_path": self.runtime_path,
            "source_release_generation": self.source_release_generation.value,
        }


@dataclass(frozen=True, slots=True)
class LegacyIdentitySchema:
    """Static catalog entry; it does not itself accept a checkpoint."""

    key: LegacyIdentityKey
    schema_id: str
    canonicalizer_id: str
    source_commit: str
    source_schema_sha256: str
    disposition: LegacyDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.key, LegacyIdentityKey):
            raise TypeError("key must be a LegacyIdentityKey")
        _require_token(self.schema_id, "schema_id")
        _require_token(self.canonicalizer_id, "canonicalizer_id")
        if (
            not isinstance(self.source_commit, str)
            or len(self.source_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.source_commit
            )
        ):
            raise ValueError("source_commit must be a full lowercase Git hash")
        if (
            not isinstance(self.source_schema_sha256, str)
            or len(self.source_schema_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_schema_sha256
            )
        ):
            raise ValueError("source_schema_sha256 must be a lowercase SHA-256")
        if not isinstance(self.disposition, LegacyDisposition):
            raise TypeError("disposition must be a LegacyDisposition")

    def to_metadata(self) -> dict[str, object]:
        return {
            "canonicalizer_id": self.canonicalizer_id,
            "disposition": self.disposition.value,
            "key": self.key.to_metadata(),
            "schema_id": self.schema_id,
            "source_commit": self.source_commit,
            "source_schema_sha256": self.source_schema_sha256,
        }


_RELEASE_COMMITS = {
    SourceReleaseGeneration.RC1: (
        "1c5237194c5e4f696bb5d3d01d2caa2c4bdd27a1"
    ),
    SourceReleaseGeneration.RC2: (
        "2bbe21e88fec318787fc31dc0a82bca5a171c024"
    ),
    SourceReleaseGeneration.RC3: (
        "5071d73a00e0d19be62ebd39edf9918818d1267a"
    ),
}

_PLANE_SCHEMA_SHA256 = (
    "c31ecb04659f7cdc18bcc43db81ddc1db2f695bfd07ed8bed08d4a9af80440d6"
)
_PERIODIC_SCHEMA_SHA256 = {
    SourceReleaseGeneration.RC1: (
        "97ac42a768925dd36a844ae02600c2e0845d1e3d1d0b7b58a1f8dc793dec33de"
    ),
    SourceReleaseGeneration.RC2: (
        "b4b70ba694c12a387bee99ebf57d5060788457e81074ecf69d7fd3b168051998"
    ),
    SourceReleaseGeneration.RC3: (
        "b4b70ba694c12a387bee99ebf57d5060788457e81074ecf69d7fd3b168051998"
    ),
}
_CHANNEL_SCHEMA_SHA256 = {
    SourceReleaseGeneration.RC1: (
        "cda03d3e854f6ee9fcb1d8ebe7b5390ecea2f06aaa6e61d590bbc983ffd00a9d"
    ),
    SourceReleaseGeneration.RC2: (
        "4a3c92fcfac6da71e869240a88fb17511857be9d37d1e73fdfef4b3181a7c456"
    ),
    SourceReleaseGeneration.RC3: (
        "4a3c92fcfac6da71e869240a88fb17511857be9d37d1e73fdfef4b3181a7c456"
    ),
}
_FUNCTIONAL_SCHEMA_SHA256 = {
    SourceReleaseGeneration.RC1: (
        "15f186833ff7b284e8fcce5899c5b5fe108e8ee0d36d0f68bd8bb29c2d8ffa43"
    ),
    SourceReleaseGeneration.RC2: (
        "9df32260b0667bd59c4328e721c4602e7ece8d43824823ea7d99683273e3cec5"
    ),
    SourceReleaseGeneration.RC3: (
        "9df32260b0667bd59c4328e721c4602e7ece8d43824823ea7d99683273e3cec5"
    ),
}


def _legacy_schema(
    family: CheckpointFamily,
    generation: SourceReleaseGeneration,
    *,
    format_version: int,
    runtime_path: str,
    schema_id: str,
    canonicalizer_id: str,
    source_schema_sha256: str,
    applicability_class: str = "configuration_dependent",
    disposition: LegacyDisposition = LegacyDisposition.ADJUDICATION_REQUIRED,
) -> LegacyIdentitySchema:
    return LegacyIdentitySchema(
        key=LegacyIdentityKey(
            checkpoint_family=family,
            format_version=format_version,
            source_release_generation=generation,
            runtime_path=runtime_path,
            applicability_class=applicability_class,
        ),
        schema_id=schema_id,
        canonicalizer_id=canonicalizer_id,
        source_commit=_RELEASE_COMMITS[generation],
        source_schema_sha256=source_schema_sha256,
        disposition=disposition,
    )


def _build_legacy_registry() -> tuple[LegacyIdentitySchema, ...]:
    entries: list[LegacyIdentitySchema] = []
    for generation in SourceReleaseGeneration:
        for runtime_path in (
            "legacy_production",
            "compiled_v2",
            "separated_canary",
        ):
            entries.append(
                _legacy_schema(
                    CheckpointFamily.PLANE_WORKFLOW,
                    generation,
                    format_version=1,
                    runtime_path=runtime_path,
                    schema_id="plane_runtime_identity_schema_v1",
                    canonicalizer_id="frozen_plane_schema_v1",
                    source_schema_sha256=_PLANE_SCHEMA_SHA256,
                )
            )
        periodic_format = 1 if generation is SourceReleaseGeneration.RC1 else 2
        functional_format = 1 if generation is SourceReleaseGeneration.RC1 else 2
        entries.extend(
            (
                _legacy_schema(
                    CheckpointFamily.PERIODIC_WORKFLOW,
                    generation,
                    format_version=periodic_format,
                    runtime_path="periodic_spectral",
                    schema_id=f"periodic_workflow_identity_v{periodic_format}",
                    canonicalizer_id=(
                        f"frozen_periodic_production_{generation.value}"
                    ),
                    source_schema_sha256=_PERIODIC_SCHEMA_SHA256[generation],
                ),
                _legacy_schema(
                    CheckpointFamily.CHANNEL_WORKFLOW,
                    generation,
                    format_version=1,
                    runtime_path="channel_complete_stress",
                    schema_id="channel_workflow_identity_v1",
                    canonicalizer_id=(
                        f"frozen_channel_production_{generation.value}"
                    ),
                    source_schema_sha256=_CHANNEL_SCHEMA_SHA256[generation],
                ),
                _legacy_schema(
                    CheckpointFamily.PERIODIC_FUNCTIONAL,
                    generation,
                    format_version=functional_format,
                    runtime_path="periodic_activity_batch_one",
                    schema_id=(
                        f"periodic_functional_identity_v{functional_format}"
                    ),
                    canonicalizer_id=(
                        f"frozen_periodic_functional_{generation.value}"
                    ),
                    source_schema_sha256=_FUNCTIONAL_SCHEMA_SHA256[generation],
                ),
                _legacy_schema(
                    CheckpointFamily.CHANNEL_FUNCTIONAL,
                    generation,
                    format_version=functional_format,
                    runtime_path="channel_activity_batch_one",
                    schema_id=(
                        f"channel_functional_identity_v{functional_format}"
                    ),
                    canonicalizer_id=(
                        f"frozen_channel_functional_{generation.value}"
                    ),
                    source_schema_sha256=_FUNCTIONAL_SCHEMA_SHA256[generation],
                ),
            )
        )
    return tuple(entries)


LEGACY_IDENTITY_SCHEMAS = _build_legacy_registry()
_LEGACY_REGISTRY = MappingProxyType(
    {entry.key: entry for entry in LEGACY_IDENTITY_SCHEMAS}
)
if len(_LEGACY_REGISTRY) != len(LEGACY_IDENTITY_SCHEMAS):
    raise RuntimeError("legacy checkpoint identity registry contains duplicate keys")


def legacy_identity_registry() -> Mapping[LegacyIdentityKey, LegacyIdentitySchema]:
    """Return the immutable registry without importing live runtime schemas."""

    return _LEGACY_REGISTRY


def resolve_legacy_identity_schema(
    key: LegacyIdentityKey,
) -> LegacyIdentitySchema:
    """Resolve an exact registry key; unknown keys fail closed."""

    if not isinstance(key, LegacyIdentityKey):
        raise TypeError("key must be a LegacyIdentityKey")
    try:
        return _LEGACY_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            "unregistered legacy checkpoint identity schema: "
            + _canonical_json(key.to_metadata(), "legacy identity key")
        ) from exc


def legacy_registry_metadata() -> dict[str, object]:
    """Return one canonical, content-addressable catalog snapshot."""

    return {
        "entries": [entry.to_metadata() for entry in LEGACY_IDENTITY_SCHEMAS],
        "entry_count": len(LEGACY_IDENTITY_SCHEMAS),
        "schema_version": LEGACY_REGISTRY_SCHEMA_VERSION,
    }


def legacy_registry_sha256() -> str:
    return _sha256_text(
        _canonical_json(legacy_registry_metadata(), "legacy identity registry")
    )


__all__ = [
    "CheckpointCompatibilityIdentity",
    "CheckpointFamily",
    "CheckpointIdentityLayer",
    "IDENTITY_SCHEMA_VERSION",
    "IdentityLayerKind",
    "LEGACY_IDENTITY_SCHEMAS",
    "LEGACY_REGISTRY_SCHEMA_VERSION",
    "LegacyDisposition",
    "LegacyIdentityKey",
    "LegacyIdentitySchema",
    "SourceReleaseGeneration",
    "legacy_identity_registry",
    "legacy_registry_metadata",
    "legacy_registry_sha256",
    "resolve_legacy_identity_schema",
]
