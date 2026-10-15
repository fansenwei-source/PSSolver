"""Stable descriptive identities for spectral-state invariants."""

from __future__ import annotations

from types import MappingProxyType


_PERIODIC_HERMITIAN_STATE_PROJECTION = MappingProxyType(
    {
        "full_complex": "complete_conjugate_pair_projection_each_step.v2",
        "hermitian_half": (
            "self_conjugate_stored_planes_projection_each_step.v2"
        ),
    }
)


def periodic_hermitian_state_projection(spectral_storage: str) -> str:
    """Return the storage-specific periodic real-spectrum repair identity."""

    try:
        return _PERIODIC_HERMITIAN_STATE_PROJECTION[spectral_storage]
    except KeyError as exc:
        raise ValueError("unsupported periodic spectral storage") from exc
