"""Unit gates for the private R7 canonical and identifier mechanics."""

from __future__ import annotations

import hashlib

import pytest

from pssolver.core._canonical import canonical_json, canonical_sha256
from pssolver.core._identifiers import (
    require_dotted_python_identifier,
    require_python_identifier,
)


def test_canonical_json_and_digest_are_compact_sorted_and_finite():
    value = {"z": [2, 1], "a": {"value": 3}}
    encoded = '{"a":{"value":3},"z":[2,1]}'
    assert canonical_json(value) == encoded
    assert canonical_sha256(value) == hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})


def test_canonical_json_preserves_caller_owned_error_message():
    with pytest.raises(
        ValueError,
        match="^identity must be finite and JSON-compatible$",
    ):
        canonical_json({"value": object()}, description="identity")


def test_identifier_helpers_preserve_caller_owned_exception_types():
    assert require_python_identifier(
        "pressure_guess",
        error=RuntimeError("unused"),
    ) == "pressure_guess"
    with pytest.raises(RuntimeError, match="^owned message$"):
        require_python_identifier("not-valid", error=RuntimeError("owned message"))

    assert require_dotted_python_identifier(
        "pssolver.runtime.builder",
        empty_error=TypeError("empty"),
        component_error=LookupError("component"),
    ) == "pssolver.runtime.builder"
    with pytest.raises(TypeError, match="^empty$"):
        require_dotted_python_identifier(
            "",
            empty_error=TypeError("empty"),
            component_error=LookupError("component"),
        )
    with pytest.raises(LookupError, match="^component$"):
        require_dotted_python_identifier(
            "pssolver.not-valid",
            empty_error=TypeError("empty"),
            component_error=LookupError("component"),
        )
