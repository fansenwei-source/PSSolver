"""Stable, machine-readable errors for the functional public API."""

from __future__ import annotations

from collections.abc import Mapping
import json

from pssolver.io.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointIOError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)
from pssolver.linear_solvers.stokes import ChannelPressureConvergenceError


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
        # Initialize the common builtin base directly.  Cooperative ``super``
        # would traverse the generic checkpoint mixin for the dual-hierarchy
        # functional checkpoint classes and overwrite their machine metadata.
        Exception.__init__(self, message)
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


class FunctionalCheckpointError(FunctionalError, CheckpointError):
    """Base class for durable functional-state failures."""

    code = "functional_checkpoint_error"

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        normalized = {
            "failure_domain": "checkpoint",
            "legacy_exception_type": None,
        }
        normalized.update(details or {})
        self.failure_domain = str(normalized["failure_domain"])
        legacy = normalized["legacy_exception_type"]
        self.legacy_exception_type = None if legacy is None else str(legacy)
        FunctionalError.__init__(
            self,
            message,
            operation=operation,
            details=normalized,
        )

    def to_metadata(self) -> dict[str, object]:
        metadata = FunctionalError.to_metadata(self)
        metadata["failure_domain"] = self.failure_domain
        metadata["legacy_exception_type"] = self.legacy_exception_type
        return metadata


class FunctionalCheckpointNotFoundError(
    FunctionalCheckpointError,
    CheckpointNotFoundError,
):
    """A required checkpoint path is missing."""

    code = "functional_checkpoint_not_found"


class FunctionalCheckpointExistsError(
    FunctionalCheckpointError,
    CheckpointExistsError,
):
    """A checkpoint export target already exists."""

    code = "functional_checkpoint_exists"


class FunctionalCheckpointIntegrityError(
    FunctionalCheckpointError,
    CheckpointIntegrityError,
):
    """Checkpoint bytes or their manifest fail integrity validation."""

    code = "functional_checkpoint_integrity_error"


class FunctionalCheckpointCompatibilityError(
    FunctionalCheckpointError,
    CheckpointCompatibilityError,
):
    """A checkpoint is intact but incompatible with the target runtime."""

    code = "functional_checkpoint_compatibility_error"


class FunctionalCheckpointIOError(
    FunctionalCheckpointError,
    CheckpointIOError,
):
    """The filesystem prevented functional checkpoint input or publication."""

    code = "functional_checkpoint_io_error"


class FunctionalExecutionError(FunctionalError, RuntimeError):
    """Functional execution failed after its inputs passed validation."""

    code = "functional_execution_error"


class FunctionalConvergenceError(
    FunctionalExecutionError,
    ChannelPressureConvergenceError,
):
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
    if isinstance(exception, ChannelPressureConvergenceError):
        return FunctionalConvergenceError(
            str(exception),
            operation=operation,
            details=exception.details,
        )
    checkpoint_details = (
        {
            **exception.details,
            "failure_domain": exception.failure_domain,
            "legacy_exception_type": exception.legacy_exception_type,
        }
        if isinstance(exception, CheckpointError)
        else None
    )
    if isinstance(exception, CheckpointNotFoundError):
        return FunctionalCheckpointNotFoundError(
            str(exception), operation=operation, details=checkpoint_details
        )
    if isinstance(exception, CheckpointExistsError):
        return FunctionalCheckpointExistsError(
            str(exception), operation=operation, details=checkpoint_details
        )
    if isinstance(exception, CheckpointCompatibilityError):
        return FunctionalCheckpointCompatibilityError(
            str(exception), operation=operation, details=checkpoint_details
        )
    if isinstance(exception, CheckpointIntegrityError):
        return FunctionalCheckpointIntegrityError(
            str(exception), operation=operation, details=checkpoint_details
        )
    if isinstance(exception, CheckpointIOError):
        return FunctionalCheckpointIOError(
            str(exception), operation=operation, details=checkpoint_details
        )
    details = {
        "failure_domain": (
            "filesystem" if isinstance(exception, OSError) else "checkpoint"
        ),
        "legacy_exception_type": type(exception).__name__,
    }
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
        return FunctionalCheckpointIOError(
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
    "FunctionalCheckpointIOError",
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
