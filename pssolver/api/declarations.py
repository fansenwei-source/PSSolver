"""Typed public conveniences that normalize to canonical declarations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pssolver.configuration.simulation import (
    ExecutionSpec,
    InitialConditionSource,
    InitialConditionSpec,
    TimeIntegrationSpec,
    WorkflowSpec,
)
from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.core.numerics import (
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    DEFAULT_SPECTRAL_STORAGE,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    DealiasRule,
    NumericsConfig,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)


def _enum(value: object, value_type: type, description: str):
    try:
        return value if isinstance(value, value_type) else value_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported {description}: {value!r}") from exc


def _positive_integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _non_negative_integer(value: object, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


class SpectralNumerics(NumericsConfig):
    """String-friendly constructor for canonical spectral numerics."""

    __slots__ = ()

    def __init__(
        self,
        *,
        dtype: str | Precision,
        dealias_rule: str | DealiasRule = DEFAULT_DEALIAS_RULE,
        transform_execution_order: str | TransformExecutionOrder = (
            DEFAULT_TRANSFORM_EXECUTION_ORDER
        ),
        projected_transform_execution: str | ProjectedTransformExecution = (
            DEFAULT_PROJECTED_TRANSFORM_EXECUTION
        ),
        spectral_storage: str | SpectralStorage = DEFAULT_SPECTRAL_STORAGE,
        hermitian_axis: int | None = None,
    ) -> None:
        super().__init__(
            precision=_enum(dtype, Precision, "dtype"),
            dealias_rule=_enum(dealias_rule, DealiasRule, "dealias rule"),
            transform_execution_order=_enum(
                transform_execution_order,
                TransformExecutionOrder,
                "transform execution order",
            ),
            projected_transform_execution=_enum(
                projected_transform_execution,
                ProjectedTransformExecution,
                "projected transform execution",
            ),
            spectral_storage=_enum(
                spectral_storage,
                SpectralStorage,
                "spectral storage",
            ),
            hermitian_axis=hermitian_axis,
        )


class TimeStepping(TimeIntegrationSpec):
    """Typed fixed-step integrator and optional refresh declaration."""

    __slots__ = ()

    def __init__(
        self,
        *,
        dt: float,
        integrator: str | IntegratorScheme = (
            IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
        ),
        refresh: Mapping[str, object] | None = None,
    ) -> None:
        scheme = _enum(integrator, IntegratorScheme, "integrator")
        if scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER:
            specification = IntegratorSpec.projected_semi_implicit_euler(dt=dt)
        else:
            specification = IntegratorSpec.sbdf2(dt=dt)
        super().__init__(
            integrator=specification,
            refresh={} if refresh is None else refresh,
        )


class GeneratedInitialCondition(InitialConditionSpec):
    """Canonical generated initial-condition request."""

    __slots__ = ()

    def __init__(
        self,
        family: str,
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            family=family,
            source=InitialConditionSource.GENERATED,
            parameters={} if parameters is None else parameters,
        )


class SnapshotInitialCondition(InitialConditionSpec):
    """Canonical external-snapshot initial-condition request."""

    __slots__ = ()

    def __init__(
        self,
        directory: str | Path,
        *,
        step: int,
        mode: str = "branch",
    ) -> None:
        if mode not in {"branch", "resume"}:
            raise ValueError("snapshot mode must be 'branch' or 'resume'")
        super().__init__(
            family="external_snapshot",
            source=InitialConditionSource.SNAPSHOT,
            parameters={
                "directory": str(Path(directory)),
                "step": _non_negative_integer(step, "snapshot step"),
                "mode": mode,
            },
        )


class TorchSpectralExecution(ExecutionSpec):
    """Explicit Torch spectral runtime request with no fallback."""

    __slots__ = ()

    def __init__(
        self,
        *,
        runtime_path: str,
        device: str,
        options: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        normalized = {} if options is None else dict(options)
        if "device" in normalized:
            raise ValueError("execution options must not repeat device")
        normalized["device"] = device
        normalized["fallback_allowed"] = False
        super().__init__(
            backend="torch_spectral",
            runtime_path=runtime_path,
            options=normalized,
        )


class Output(WorkflowSpec):
    """Finite output, observation, checkpoint, and restart schedule."""

    __slots__ = ()

    def __init__(
        self,
        *,
        directory: str | Path,
        steps: int,
        save_interval: int,
        diagnostic_interval: int,
        save_start_step: int = 0,
        diagnostics: bool = True,
        save_hydrodynamics: bool = True,
        checkpoint_interval: int | None = None,
        restart_from: str | Path | None = None,
        options: Mapping[str, object] | None = None,
    ) -> None:
        steps = _positive_integer(steps, "steps")
        save_start_step = _non_negative_integer(
            save_start_step,
            "save_start_step",
        )
        if save_start_step > steps:
            raise ValueError("save_start_step must not exceed steps")
        values = {} if options is None else dict(options)
        reserved = {
            "output_dir",
            "save_start_step",
            "save_interval",
            "diagnostic_interval",
            "diagnostics",
            "save_hydrodynamics",
            "checkpoint_interval",
            "restart_from",
        }
        overlap = reserved & set(values)
        if overlap:
            raise ValueError(
                "output options repeat typed fields: "
                f"{tuple(sorted(overlap))!r}"
            )
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be a bool")
        if not isinstance(save_hydrodynamics, bool):
            raise TypeError("save_hydrodynamics must be a bool")
        if checkpoint_interval is not None:
            checkpoint_interval = _positive_integer(
                checkpoint_interval,
                "checkpoint_interval",
            )
        values.update(
            {
                "output_dir": str(Path(directory)),
                "save_start_step": save_start_step,
                "save_interval": _positive_integer(
                    save_interval,
                    "save_interval",
                ),
                "diagnostic_interval": _positive_integer(
                    diagnostic_interval,
                    "diagnostic_interval",
                ),
                "diagnostics": diagnostics,
                "save_hydrodynamics": save_hydrodynamics,
                "checkpoint_interval": checkpoint_interval,
                "restart_from": (
                    None if restart_from is None else str(Path(restart_from))
                ),
            }
        )
        super().__init__(steps=steps, options=values)


__all__ = [
    "GeneratedInitialCondition",
    "Output",
    "SnapshotInitialCondition",
    "SpectralNumerics",
    "TimeStepping",
    "TorchSpectralExecution",
]
