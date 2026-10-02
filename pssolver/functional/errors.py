"""Stable, machine-readable errors for the functional public API."""

from __future__ import annotations

from collections.abc import Mapping
import json


class FunctionalError(Exception):
    """Base class for failures crossing the stable functional boundary."""

    code = "functional_error"

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.details = _json_copy(details or {})

    def to_metadata(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "operation": self.operation,
            "details": dict(self.details),
        }


class FunctionalContractError(FunctionalError):
    """The caller supplied a value outside the public contract."""

    code = "functional_contract_error"


class FunctionalTypeError(FunctionalContractError, TypeError):
    """A public functional value has the wrong Python or tensor type."""

    code = "functional_type_error"


class FunctionalValueError(FunctionalContractError, ValueError):
    """A public functional value violates shape, device, or value rules."""

    code = "functional_value_error"


class FunctionalVersionError(FunctionalValueError):
    """The requested protocol version is unsupported for the operation."""

    code = "functional_version_error"


class FunctionalIdentityError(FunctionalValueError):
    """A runtime, layout, backend, or scientific identity does not match."""

    code = "functional_identity_error"


class FunctionalCheckpointError(FunctionalError):
    """Base class for durable functional-state failures."""

    code = "functional_checkpoint_error"


class FunctionalCheckpointNotFoundError(
    FunctionalCheckpointError,
    FileNotFoundError,
):
    """A required checkpoint path is missing."""

    code = "functional_checkpoint_not_found"


class FunctionalCheckpointExistsError(
    FunctionalCheckpointError,
    FileExistsError,
):
    """A checkpoint export target already exists."""

    code = "functional_checkpoint_exists"


class FunctionalCheckpointIntegrityError(
    FunctionalCheckpointError,
    ValueError,
):
    """Checkpoint bytes or their manifest fail integrity validation."""

    code = "functional_checkpoint_integrity_error"


class FunctionalCheckpointCompatibilityError(
    FunctionalCheckpointError,
    ValueError,
):
    """A checkpoint is intact but incompatible with the target runtime."""

    code = "functional_checkpoint_compatibility_error"


class FunctionalExecutionError(FunctionalError, RuntimeError):
    """Functional execution failed after its inputs passed validation."""

    code = "functional_execution_error"


class FunctionalConvergenceError(FunctionalExecutionError):
    """An inner solve did not satisfy its declared convergence contract."""

    code = "functional_convergence_error"


def translate_functional_exception(
    exception: Exception,
    *,
    operation: str,
) -> FunctionalError:
    """Map compatibility-path built-ins without inspecting message text."""

    if isinstance(exception, FunctionalError):
        return exception
    details = {"legacy_exception_type": type(exception).__name__}
    if isinstance(exception, FileNotFoundError):
        return FunctionalCheckpointNotFoundError(
            str(exception), operation=operation, details=details
        )
    if isinstance(exception, FileExistsError):
        return FunctionalCheckpointExistsError(
            str(exception), operation=operation, details=details
        )
    if operation in {"checkpoint_export", "checkpoint_import"} and isinstance(
        exception,
        OSError,
    ):
        return FunctionalCheckpointError(
            str(exception), operation=operation, details=details
        )
    if operation == "checkpoint_import" and isinstance(
        exception,
        (TypeError, ValueError),
    ):
        return FunctionalCheckpointIntegrityError(
            str(exception), operation=operation, details=details
        )
    if isinstance(exception, TypeError):
        return FunctionalTypeError(
            str(exception), operation=operation, details=details
        )
    if isinstance(exception, ValueError):
        return FunctionalValueError(
            str(exception), operation=operation, details=details
        )
    if isinstance(exception, RuntimeError):
        return FunctionalExecutionError(
            str(exception), operation=operation, details=details
        )
    return FunctionalError(str(exception), operation=operation, details=details)


def _json_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    try:
        return json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("functional error details must be JSON-compatible") from exc


__all__ = [
    "FunctionalCheckpointCompatibilityError",
    "FunctionalCheckpointError",
    "FunctionalCheckpointExistsError",
    "FunctionalCheckpointIntegrityError",
    "FunctionalCheckpointNotFoundError",
    "FunctionalContractError",
    "FunctionalConvergenceError",
    "FunctionalError",
    "FunctionalExecutionError",
    "FunctionalIdentityError",
    "FunctionalTypeError",
    "FunctionalValueError",
    "FunctionalVersionError",
    "translate_functional_exception",
]
