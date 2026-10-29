"""Private, standard-library-only canonical JSON identity helpers.

The functions in this module deliberately implement only the compact JSON
encoding used for runtime identities.  Pretty-printed checkpoint documents
and file digests retain their format-specific owners.
"""

from __future__ import annotations

import hashlib
import json


def canonical_json(value: object, *, description: str | None = None) -> str:
    """Return the compact, finite, key-sorted JSON identity encoding.

    ``description`` lets policy-owning callers preserve their historical
    public ``ValueError`` message while sharing the serialization mechanic.
    When it is omitted, the original ``json`` exception is propagated.
    """

    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        if description is None:
            raise
        raise ValueError(
            f"{description} must be finite and JSON-compatible"
        ) from exc


def canonical_sha256(
    value: object,
    *,
    description: str | None = None,
) -> str:
    """Return SHA-256 of :func:`canonical_json` encoded as UTF-8."""

    payload = canonical_json(value, description=description).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["canonical_json", "canonical_sha256"]
