"""Read compact-Q initial snapshots with the shared input-error contract."""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from pssolver.core.errors import (
    SimulationInputError,
    SimulationInputFileNotFoundError,
    SimulationInputIOError,
)


def snapshot_sha256(path: Path) -> str:
    """Hash the original snapshot file, not its decoded tensor."""
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reject_snapshot_input(
    message: str,
    *,
    application_family: str,
    path: Path,
    reason: str,
    exception: Exception | None = None,
) -> NoReturn:
    """Raise the established initial-condition error with its source context."""
    error_type = (
        SimulationInputFileNotFoundError
        if isinstance(exception, FileNotFoundError)
        else SimulationInputIOError
        if isinstance(exception, (OSError, ValueError))
        else SimulationInputError
    )
    error = error_type(
        message,
        application_family=application_family,
        stage="initial_condition",
        step=0,
        details={
            "path": str(path),
            "reason": reason,
            "legacy_exception_type": None if exception is None else type(exception).__name__,
        },
    )
    if exception is None:
        raise error
    raise error from exception


def load_q_snapshot(
    directory: str | Path,
    step: int,
    *,
    shape: tuple[int, ...],
    dtype: str,
    components: Sequence[str],
    label: str,
    application_family: str,
) -> tuple[dict[str, torch.Tensor], Path, str]:
    """Load compact Q and return CPU components, path and original file digest.

    Either component-first or component-last arrays are accepted, with the
    latter taking precedence if the two shapes coincide. The output tensor for
    each named component has exactly the declared grid shape and dtype.
    """
    path = Path(directory).expanduser().resolve() / f"Q_{step}.npy"

    def reject(message: str, reason: str, exception: Exception | None = None) -> NoReturn:
        reject_snapshot_input(
            f"{label} Q snapshot {message}",
            application_family=application_family,
            path=path,
            reason=reason,
            exception=exception,
        )

    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        reject("could not be read", "snapshot_unreadable", exc)
    if values.shape == (*shape, len(components)):
        values = np.moveaxis(values, -1, 0)
    elif values.shape != (len(components), *shape):
        reject("must have shape (*grid, 5) or (5, *grid)", "snapshot_shape_mismatch")
    if str(values.dtype) != dtype:
        reject(f"dtype must be {dtype}, got {values.dtype}", "snapshot_dtype_mismatch")
    if not np.isfinite(values).all():
        reject("contains NaN or Inf", "snapshot_nonfinite")
    tensors = torch.from_numpy(np.array(values, copy=True))
    try:
        digest = snapshot_sha256(path)
    except OSError as exc:
        reject("could not be hashed", "snapshot_unreadable", exc)
    return {name: tensors[index] for index, name in enumerate(components)}, path, digest
