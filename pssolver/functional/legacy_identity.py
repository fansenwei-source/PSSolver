"""Frozen functional-runtime identity schemas from released checkpoints.

The stable functional API kept the public ``api_version`` literal at ``1.0``
while its runtime identity acquired additional, qualified fields.  Legacy
checkpoint validation must therefore rebuild the identity schema owned by the
source release.  It must not hash today's complete identity and merely change
an API literal or remove whichever field most recently changed.

These canonicalizers deliberately enumerate every release-owned
``functional_runtime`` field.  Additions to the current runtime identity are
ignored unless a future release receives its own frozen canonicalizer.
Scientific, discretization, base execution, and state-layout identities remain
configuration dependent and are copied from the target construction.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Literal

from pssolver.core._canonical import canonical_json


LegacyFunctionalFamily = Literal["periodic", "channel"]
LegacyReleaseGeneration = Literal["rc1", "rc2", "rc3"]

_RELEASE_GENERATIONS = frozenset({"rc1", "rc2", "rc3"})
_LEGACY_FUNCTIONAL_API_VERSION = "1.0"
_CURRENT_FUNCTIONAL_API_VERSION = "1.1"


def _json_copy(value: object, description: str) -> object:
    try:
        return json.loads(
            canonical_json(value)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be JSON-compatible") from exc


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    copied = _json_copy(dict(value), description)
    if not isinstance(copied, dict):  # pragma: no cover - guarded above
        raise AssertionError("JSON mapping copy changed type")
    return copied


def _required(mapping: Mapping[str, object], key: str, description: str) -> object:
    if key not in mapping:
        raise ValueError(f"{description} is missing {key}")
    return mapping[key]


def _periodic_runtime_identity(
    current: Mapping[str, object],
) -> dict[str, object]:
    if current.get("kind") != "periodic_activity_batch_one":
        raise ValueError("Periodic functional runtime kind is invalid")
    fields = (
        "device",
        "pointwise_execution",
        "q_gradient_reuse",
        "pressure_solver",
        "spectral_refresh",
        "activity_product_before_divergence",
        "snapshot_sha256",
        "fallback_allowed",
        "fallback_used",
        "deterministic_replay",
        "durable_checkpoint_bridge_version",
    )
    return {
        "kind": "periodic_activity_batch_one",
        **{
            key: _json_copy(
                _required(current, key, "Periodic functional runtime"),
                f"Periodic functional runtime {key}",
            )
            for key in fields
        },
        # rc1--rc3 all wrote this storage-agnostic description.  Keep the
        # released literal frozen after R5-11 corrected current metadata.
        "hermitian_state_projection": "self_conjugate_planes_each_step",
    }


def _channel_pressure_identity(
    current: Mapping[str, object],
    generation: LegacyReleaseGeneration,
) -> dict[str, object]:
    if current.get("kind") != "channel_no_slip_modal_stokes_pcg":
        raise ValueError("Channel pressure runtime kind is invalid")
    fixed_iterations = _required(
        current,
        "fixed_iterations",
        "Channel pressure runtime",
    )
    result = {
        "kind": "channel_no_slip_modal_stokes_pcg",
        "algorithm": _json_copy(
            _required(current, "algorithm", "Channel pressure runtime"),
            "Channel pressure algorithm",
        ),
        "relative_tolerance": _json_copy(
            _required(
                current,
                "relative_tolerance",
                "Channel pressure runtime",
            ),
            "Channel pressure relative tolerance",
        ),
        "max_iterations": _json_copy(
            _required(current, "max_iterations", "Channel pressure runtime"),
            "Channel pressure maximum iterations",
        ),
        "fixed_iterations": _json_copy(
            fixed_iterations,
            "Channel pressure fixed iterations",
        ),
    }
    if generation != "rc1":
        result["stopping_precedence"] = (
            "fixed_iterations_overrides_max_iterations;"
            "relative_tolerance_is_early_exit_only"
            if fixed_iterations is not None
            else "relative_tolerance_required_with_max_iterations_cap"
        )
    result.update(
        {
            "gauge": "zero_mean",
            "initial_guess": "zero_every_call",
            "production_warm_start_read": False,
            "production_warm_start_written": False,
            "gradient": "custom_implicit_pressure_adjoint_v1",
        }
    )
    return result


def _channel_runtime_identity(
    current: Mapping[str, object],
    generation: LegacyReleaseGeneration,
) -> dict[str, object]:
    if current.get("kind") != "channel_activity_batch_one":
        raise ValueError("Channel functional runtime kind is invalid")
    pressure = _mapping(
        _required(current, "pressure_solver", "Channel functional runtime"),
        "Channel pressure runtime",
    )
    return {
        "kind": "channel_activity_batch_one",
        "stage": "P9.7.4",
        "device": _json_copy(
            _required(current, "device", "Channel functional runtime"),
            "Channel functional device",
        ),
        "pointwise_execution": "eager",
        "q_gradient_reuse": False,
        "spectral_refresh": "disabled",
        "activity_product_before_divergence": True,
        "pressure_solver": _channel_pressure_identity(pressure, generation),
        "boundary_spaces": _json_copy(
            _required(current, "boundary_spaces", "Channel functional runtime"),
            "Channel boundary spaces",
        ),
        "snapshot_sha256": _json_copy(
            _required(current, "snapshot_sha256", "Channel functional runtime"),
            "Channel snapshot SHA-256",
        ),
        "fallback_allowed": False,
        "fallback_used": False,
        "deterministic_replay": "bitwise",
        "durable_checkpoint_bridge_version": 1,
        "checkpoint_export": "functional_q_state_only",
        "production_checkpoint_import": "q_state_only_after_identity_gate",
    }


def legacy_runtime_identity_metadata(
    metadata: Mapping[str, object],
    *,
    family: LegacyFunctionalFamily,
    source_release_generation: LegacyReleaseGeneration,
) -> dict[str, object]:
    """Return the exact runtime-identity payload owned by rc1, rc2, or rc3."""

    if family not in {"periodic", "channel"}:
        raise ValueError("legacy functional family is invalid")
    if source_release_generation not in _RELEASE_GENERATIONS:
        raise ValueError("source release generation is invalid")
    payload = _mapping(metadata, "functional runtime identity")
    if set(payload) != {
        "api_version",
        "scientific",
        "discretization",
        "execution",
        "state_layout",
    }:
        raise ValueError("functional runtime identity top-level schema is invalid")
    if payload["api_version"] not in {
        _LEGACY_FUNCTIONAL_API_VERSION,
        _CURRENT_FUNCTIONAL_API_VERSION,
    }:
        raise ValueError(
            "functional API version cannot be projected to the released "
            "1.0 identity schema"
        )
    execution = _mapping(payload["execution"], "functional execution identity")
    runtime = _mapping(
        _required(execution, "functional_runtime", "functional execution identity"),
        "functional runtime execution identity",
    )
    execution["functional_runtime"] = (
        _periodic_runtime_identity(runtime)
        if family == "periodic"
        else _channel_runtime_identity(runtime, source_release_generation)
    )
    return {
        "api_version": _LEGACY_FUNCTIONAL_API_VERSION,
        "scientific": payload["scientific"],
        "discretization": payload["discretization"],
        "execution": execution,
        "state_layout": payload["state_layout"],
    }


def legacy_runtime_identity_sha256(
    metadata: Mapping[str, object],
    *,
    family: LegacyFunctionalFamily,
    source_release_generation: LegacyReleaseGeneration,
) -> str:
    """Hash one release-owned, configuration-dependent legacy identity."""

    payload = legacy_runtime_identity_metadata(
        metadata,
        family=family,
        source_release_generation=source_release_generation,
    )
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "LegacyFunctionalFamily",
    "LegacyReleaseGeneration",
    "legacy_runtime_identity_metadata",
    "legacy_runtime_identity_sha256",
]
