"""Version negotiation and provenance for the stable functional protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

from pssolver._version import __version__ as PSSOLVER_PACKAGE_VERSION

from .errors import FunctionalVersionError


FUNCTIONAL_API_VERSION = "1.0"
FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS = ("0.1-provisional",)
FUNCTIONAL_CONSTRUCTION_API_VERSIONS = (FUNCTIONAL_API_VERSION,)
FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS = (
    FUNCTIONAL_API_VERSION,
    *FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS,
)
FUNCTIONAL_COMPATIBILITY_POLICY_VERSION = 1

FunctionalVersionPurpose = Literal["construction", "checkpoint_read"]


@dataclass(frozen=True, slots=True)
class FunctionalAPIVersionSelection:
    """Exact result of one fail-closed protocol-version negotiation."""

    requested: str
    effective: str
    purpose: FunctionalVersionPurpose
    compatibility_reader: str | None

    @property
    def legacy(self) -> bool:
        return self.requested != self.effective

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "purpose": self.purpose,
            "legacy": self.legacy,
            "compatibility_reader": self.compatibility_reader,
        }


def negotiate_functional_api_version(
    requested: str,
    *,
    purpose: FunctionalVersionPurpose,
) -> FunctionalAPIVersionSelection:
    """Negotiate by exact literal; only checkpoints receive legacy readers."""

    if not isinstance(requested, str):
        raise FunctionalVersionError(
            "functional API version must be a string",
            operation="negotiate_functional_api_version",
            details={"purpose": purpose},
        )
    if purpose == "construction":
        if requested != FUNCTIONAL_API_VERSION:
            raise FunctionalVersionError(
                "functional construction API version is unsupported",
                operation="negotiate_functional_api_version",
                details={
                    "requested": requested,
                    "purpose": purpose,
                    "supported": list(FUNCTIONAL_CONSTRUCTION_API_VERSIONS),
                },
            )
        return FunctionalAPIVersionSelection(
            requested=requested,
            effective=FUNCTIONAL_API_VERSION,
            purpose=purpose,
            compatibility_reader=None,
        )
    if purpose == "checkpoint_read":
        if requested == FUNCTIONAL_API_VERSION:
            return FunctionalAPIVersionSelection(
                requested=requested,
                effective=FUNCTIONAL_API_VERSION,
                purpose=purpose,
                compatibility_reader=None,
            )
        if requested in FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS:
            return FunctionalAPIVersionSelection(
                requested=requested,
                effective=FUNCTIONAL_API_VERSION,
                purpose=purpose,
                compatibility_reader=(
                    "qualified_v1_exact_schema_reader_for_"
                    + requested.replace(".", "_").replace("-", "_")
                ),
            )
        raise FunctionalVersionError(
            "functional checkpoint API version is unsupported",
            operation="negotiate_functional_api_version",
            details={
                "requested": requested,
                "purpose": purpose,
                "supported": list(FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS),
            },
        )
    raise FunctionalVersionError(
        "functional API version purpose is unsupported",
        operation="negotiate_functional_api_version",
        details={"purpose": purpose},
    )


def runtime_identity_sha256_for_api_version(
    metadata: dict[str, object],
    api_version: str,
) -> str:
    """Hash an identity under a selected, already-negotiated API literal."""

    selection = negotiate_functional_api_version(
        api_version,
        purpose="checkpoint_read",
    )
    payload = dict(metadata)
    payload["api_version"] = selection.requested
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def functional_protocol_provenance() -> dict[str, object]:
    """Return package, protocol, compatibility, and deprecation provenance."""

    return {
        "schema_version": 1,
        "package": {
            "name": "pssolver",
            "version": PSSOLVER_PACKAGE_VERSION,
        },
        "functional_protocol": {
            "current": FUNCTIONAL_API_VERSION,
            "construction_versions": list(
                FUNCTIONAL_CONSTRUCTION_API_VERSIONS
            ),
            "checkpoint_read_versions": list(
                FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS
            ),
            "negotiation": "exact_literal_fail_closed",
            "silent_fallback": False,
        },
        "compatibility_policy": {
            "version": FUNCTIONAL_COMPATIBILITY_POLICY_VERSION,
            "legacy_checkpoint_schema_repair": False,
            "source_shadowing_allowed": False,
            "FunctionalCapabilities_alias": {
                "status": "compatibility_public",
                "canonical_name": "FunctionalCapabilitySet",
                "deprecated_in_package_version": "0.2.0",
                "earliest_removal_package_version": "0.4.0",
            },
        },
    }


__all__ = [
    "FUNCTIONAL_API_VERSION",
    "FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS",
    "FUNCTIONAL_COMPATIBILITY_POLICY_VERSION",
    "FUNCTIONAL_CONSTRUCTION_API_VERSIONS",
    "FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS",
    "FunctionalAPIVersionSelection",
    "FunctionalVersionPurpose",
    "functional_protocol_provenance",
    "negotiate_functional_api_version",
    "runtime_identity_sha256_for_api_version",
]
