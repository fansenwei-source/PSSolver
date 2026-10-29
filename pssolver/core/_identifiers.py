"""Private, tensor-free Python identifier validation mechanics."""

from __future__ import annotations


def require_python_identifier(value: object, *, error: Exception) -> str:
    """Return one identifier or raise the caller-owned public exception."""

    if not isinstance(value, str) or not value.isidentifier():
        raise error
    return value


def require_dotted_python_identifier(
    value: object,
    *,
    empty_error: Exception,
    component_error: Exception,
) -> str:
    """Return a nonempty dotted identifier with caller-owned errors."""

    if not isinstance(value, str) or not value:
        raise empty_error
    if any(not part.isidentifier() for part in value.split(".")):
        raise component_error
    return value


__all__ = ["require_dotted_python_identifier", "require_python_identifier"]
