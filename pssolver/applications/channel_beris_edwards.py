"""P8.3 production application for complete-stress rectangular Channels."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pssolver.errors import (
    SimulationInputError,
    SimulationInputFileNotFoundError,
    SimulationInputIOError,
    SimulationRestartError,
)
from pssolver.io.checkpoint import CheckpointError

from pssolver.configuration.channel_beris_edwards import (
    ChannelBerisEdwardsRunSpec,
)
from pssolver.configuration.execution_device import (
    configure_tf32_execution,
    execution_device_normalization,
    resolve_execution_device,
    scoped_tf32_execution,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.workflows.channel_beris_edwards import (
    ChannelBerisEdwardsWorkflow,
    read_channel_beris_edwards_checkpoint_progress,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_input(
    message: str,
    *,
    path: Path,
    reason: str,
    exception: Exception | None = None,
) -> None:
    error_type = (
        SimulationInputFileNotFoundError
        if isinstance(exception, FileNotFoundError)
        else SimulationInputIOError
        if isinstance(exception, (OSError, ValueError))
        else SimulationInputError
    )
    error = error_type(
        message,
        application_family="channel_complete_stress_beris_edwards",
        stage="initial_condition",
        step=0,
        details={
            "path": str(path),
            "reason": reason,
            "legacy_exception_type": (
                None if exception is None else type(exception).__name__
            ),
        },
    )
    if exception is None:
        raise error
    raise error from exception


def _reject_restart(
    message: str,
    *,
    reason: str,
    step: int = 0,
    details: dict[str, object] | None = None,
    exception: Exception | None = None,
) -> None:
    error = SimulationRestartError(
        message,
        application_family="channel_complete_stress_beris_edwards",
        stage="restart_validation",
        step=step,
        details={"reason": reason, **(details or {})},
    )
    if exception is None:
        raise error
    raise error from exception


def load_channel_initial_q(run_spec: ChannelBerisEdwardsRunSpec):
    """Load the canonical Channel Q snapshot without constructing a runtime."""

    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a ChannelBerisEdwardsRunSpec")
    spec = run_spec.simulation
    snapshot = spec.initial_condition.parameters
    path = Path(snapshot["directory"]).expanduser().resolve() / (
        f"Q_{snapshot['step']}.npy"
    )
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        _reject_input(
            "Channel Q snapshot could not be read",
            path=path,
            reason="snapshot_unreadable",
            exception=exc,
        )
    shape = tuple(spec.geometry.domain.shape)
    if values.shape == (*shape, len(Q_COMPONENTS)):
        values = np.moveaxis(values, -1, 0)
    elif values.shape != (len(Q_COMPONENTS), *shape):
        _reject_input(
            "Channel Q snapshot must have shape (*grid, 5) or (5, *grid)",
            path=path,
            reason="snapshot_shape_mismatch",
        )
    expected_dtype = spec.numerics.precision.value
    if str(values.dtype) != expected_dtype:
        _reject_input(
            f"Channel Q snapshot dtype must be {expected_dtype}, got {values.dtype}",
            path=path,
            reason="snapshot_dtype_mismatch",
        )
    if not np.isfinite(values).all():
        _reject_input(
            "Channel Q snapshot contains NaN or Inf",
            path=path,
            reason="snapshot_nonfinite",
        )
    tensors = torch.from_numpy(np.array(values, copy=True))
    try:
        snapshot_sha256 = _sha256(path)
    except OSError as exc:
        _reject_input(
            "Channel Q snapshot could not be hashed",
            path=path,
            reason="snapshot_unreadable",
            exception=exc,
        )
    return (
        {name: tensors[index] for index, name in enumerate(Q_COMPONENTS)},
        path,
        snapshot_sha256,
    )


def _run_channel_beris_edwards_owned(
    run_spec: ChannelBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    tf32_metadata: dict[str, object],
):
    """Execute one frozen P8.3 application with no runtime fallback."""

    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be ChannelBerisEdwardsRunSpec")
    spec = run_spec.simulation
    metadata = {
        "schema_version": 1,
        "application": "channel_complete_stress_beris_edwards",
        "configuration": spec.to_metadata(),
        "configuration_sha256": spec.canonical_sha256(),
        "runtime_identity_sha256": run_spec.runtime_identity_sha256(),
        "runtime_selection": {
            "requested": run_spec.runtime_path,
            "effective": run_spec.runtime_path,
            "fallback_allowed": False,
            "fallback_used": False,
        },
        "normalization": execution_device_normalization(
            spec.execution.options["device"]
        ),
        "pressure_gauge": "zero_mean",
        "velocity_nullspace": "absent_due_to_two_bounded_no_slip_axes",
    }
    if emit_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    if run_spec.dry_run:
        return None

    device = resolve_execution_device(spec.execution.options["device"])
    real_dtype = getattr(torch, spec.numerics.precision.value)
    metadata["precision"] = {
        "real_dtype": spec.numerics.precision.value,
        "spectral_dtype": (
            "complex64" if real_dtype == torch.float32 else "complex128"
        ),
        **tf32_metadata,
    }
    restart = spec.workflow.options["restart_from"]
    if restart is not None:
        try:
            start = read_channel_beris_edwards_checkpoint_progress(restart)
        except CheckpointError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            _reject_restart(
                str(exc) or "Channel checkpoint preflight failed",
                reason="checkpoint_preflight_failed",
                details={"legacy_exception_type": type(exc).__name__},
                exception=exc,
            )
        if spec.workflow.options["save_start_step"] > start + spec.workflow.steps:
            _reject_restart(
                "save_start_step must not exceed the absolute final step",
                reason="save_start_step_after_final_step",
                step=start,
                details={
                    "save_start_step": spec.workflow.options["save_start_step"],
                    "absolute_final_step": start + spec.workflow.steps,
                },
            )
    initial_values, snapshot_path, snapshot_sha256 = load_channel_initial_q(
        run_spec
    )
    metadata["initial_condition"] = {
        "source": str(snapshot_path),
        "sha256": snapshot_sha256,
        "logical_mode": spec.initial_condition.parameters["mode"],
    }
    output = run_spec.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty Channel output: {output}")
    request = ChannelBerisEdwardsRuntimeBuildRequest(
        run_spec=run_spec,
        initial_values=initial_values,
        device=device,
    )
    construction = PackageRuntimeConstructionInput(
        plan=plan_package_runtime_construction(spec),
        request=request,
    )
    adapter = build_package_simulation_runtime(construction)
    metadata["runtime_construction"] = construction.to_metadata()
    metadata["runtime"] = adapter.to_metadata()
    workflow = ChannelBerisEdwardsWorkflow(
        adapter,
        run_spec,
        output,
        metadata,
    )
    return workflow.run(progress=progress)


def run_channel_beris_edwards(
    run_spec: ChannelBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
):
    """Execute one Channel run inside a dynamically scoped TF32 policy."""

    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be ChannelBerisEdwardsRunSpec")
    spec = run_spec.simulation
    real_dtype = getattr(torch, spec.numerics.precision.value)
    requested = spec.execution.options["tf32"]
    if run_spec.dry_run:
        return _run_channel_beris_edwards_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            tf32_metadata=configure_tf32_execution(
                requested,
                real_dtype=real_dtype,
                device=spec.execution.options["device"],
            ),
        )
    device = resolve_execution_device(spec.execution.options["device"])
    with scoped_tf32_execution(
        requested,
        real_dtype=real_dtype,
        device=device,
    ) as tf32_metadata:
        return _run_channel_beris_edwards_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            tf32_metadata=tf32_metadata,
        )


__all__ = ["load_channel_initial_q", "run_channel_beris_edwards"]
