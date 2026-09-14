"""Shared stepping, observation, diagnostics, and restart for Plane runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

from pssolver.configuration import PlaneBerisEdwardsRunSpec
from pssolver.run_metadata import write_run_metadata
from pssolver.runtime import PlaneRuntimeAdapterProtocol

from .plane_checkpoint import (
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    capture_plane_checkpoint,
    load_plane_checkpoint,
    restore_plane_checkpoint,
    write_plane_checkpoint,
)
from .plane_observation import (
    PlaneDiagnostic,
    PlaneObservation,
    capture_plane_diagnostic,
    capture_plane_observation,
    write_plane_diagnostics,
    write_plane_observation,
)


PLANE_WORKFLOW_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PlaneWorkflowResult:
    """Completion record returned only after the marker is written last."""

    start_step: int
    final_step: int
    elapsed_seconds: float
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_observation: PlaneObservation
    diagnostics: tuple[PlaneDiagnostic, ...]


class PlaneBerisEdwardsWorkflow:
    """Backend-neutral coordinator for the current Plane output contract."""

    def __init__(
        self,
        adapter: PlaneRuntimeAdapterProtocol,
        run_spec: PlaneBerisEdwardsRunSpec,
        output_directory: str | Path,
        metadata: Mapping[str, object],
    ) -> None:
        if not isinstance(adapter, PlaneRuntimeAdapterProtocol):
            raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")
        if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
            raise TypeError("run_spec must be PlaneBerisEdwardsRunSpec")
        if adapter.runtime_path is not run_spec.runtime_path:
            raise ValueError("workflow runtime path differs from run specification")
        directory = Path(output_directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Plane output directory is missing: {directory}")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        self.adapter = adapter
        self.run_spec = run_spec
        self.output_directory = directory
        self.metadata = deepcopy(dict(metadata))
        self._diagnostics: list[PlaneDiagnostic] = []
        self._saved_steps: list[int] = []
        self._checkpoint_steps: list[int] = []

    def _capture_diagnostic(self, step: int) -> PlaneDiagnostic:
        value = capture_plane_diagnostic(
            self.adapter,
            step=step,
            viscosity=self.run_spec.eta,
            friction=(
                0.0
                if self.run_spec.zero_mode_policy == "zero_mean"
                else self.run_spec.friction_mode_fric
            ),
        )
        self._diagnostics.append(value)
        return value

    def _save_observation(self, step: int) -> PlaneObservation:
        observation = capture_plane_observation(self.adapter, step=step)
        write_plane_observation(
            self.output_directory,
            observation,
            save_hydrodynamics=self.run_spec.save_hydrodynamics,
        )
        self._saved_steps.append(step)
        return observation

    def _save_checkpoint(self) -> None:
        checkpoint = capture_plane_checkpoint(
            self.adapter,
            runtime_identity_sha256=(
                self.run_spec.runtime_identity_sha256()
            ),
        )
        write_plane_checkpoint(
            self.output_directory / f"checkpoint_{checkpoint.completed_steps}",
            checkpoint,
        )
        self._checkpoint_steps.append(checkpoint.completed_steps)

    def _restore_if_requested(self) -> int:
        source = self.run_spec.restart_from
        if source is None:
            if self.adapter.completed_steps != 0:
                raise ValueError("new Plane workflow requires an unadvanced runtime")
            self.metadata["restart"] = {"kind": "fresh"}
            return 0
        checkpoint = load_plane_checkpoint(source)
        start = restore_plane_checkpoint(
            self.adapter,
            checkpoint,
            runtime_identity_sha256=self.run_spec.runtime_identity_sha256(),
        )
        self.metadata["restart"] = {
            "kind": "same_backend_plane_workflow_checkpoint",
            "source": str(Path(source).expanduser().resolve()),
            "source_runtime_path": checkpoint.runtime_path.value,
            "source_runtime_identity_sha256": (
                checkpoint.runtime_identity_sha256
            ),
            "source_completed_steps": checkpoint.completed_steps,
            "cross_backend_adapter_used": False,
        }
        self.metadata["initial_condition"][
            "used_for_runtime_construction_only"
        ] = True
        return start

    def _record_pre_update(self, step: int, progress: object) -> None:
        if (
            self.run_spec.diagnostics
            and step % self.run_spec.diagnostic_interval == 0
        ):
            value = self._capture_diagnostic(step)
            setter = getattr(progress, "set_postfix", None)
            if callable(setter):
                setter(
                    div_max=f"{value.div_max:.2e}",
                    div_rms=f"{value.div_rms:.2e}",
                    div_rel=f"{value.div_rel:.2e}",
                    schur_it=int(value.schur_iterations),
                    schur_rel=f"{value.schur_rel_residual:.2e}",
                    wall_n_rms=(
                        f"{value.wall_normal_momentum_rms:.2e}"
                    ),
                )
        if (
            step >= self.run_spec.save_start_step
            and step % self.run_spec.save_interval == 0
        ):
            self._save_observation(step)

    def run(
        self,
        *,
        progress: Iterable[int] | None = None,
    ) -> PlaneWorkflowResult:
        """Run the requested additional steps and complete atomically by order."""

        if (self.output_directory / "COMPLETE").exists():
            raise FileExistsError("Plane output directory is already complete")
        start_step = self._restore_if_requested()
        self.metadata["workflow"] = {
            "schema_version": PLANE_WORKFLOW_SCHEMA_VERSION,
            "runtime_path": self.adapter.runtime_path.value,
            "checkpoint_format_version": (
                PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION
            ),
            "cross_backend_restart_supported": False,
            "start_step": start_step,
            "requested_additional_steps": self.run_spec.steps,
            "completion_marker_order": "metadata_then_COMPLETE",
        }
        write_run_metadata(
            self.output_directory,
            self.metadata,
            status="running",
        )

        iterator = range(self.run_spec.steps) if progress is None else progress
        executed = 0
        started = time.time()
        for local_step in iterator:
            if local_step != executed:
                raise ValueError("progress iterator must yield consecutive local steps")
            absolute_step = start_step + local_step
            self.adapter.advance(
                1,
                pre_update_callback=(
                    lambda _solver, _step, absolute_step=absolute_step: (
                        self._record_pre_update(absolute_step, iterator)
                    )
                ),
            )
            executed += 1
            if self.adapter.completed_steps != start_step + executed:
                raise RuntimeError("runtime completed-step clock diverged")
            interval = self.run_spec.checkpoint_interval
            if interval is not None and self.adapter.completed_steps % interval == 0:
                self._save_checkpoint()
        if executed != self.run_spec.steps:
            raise ValueError("progress iterator length differs from requested steps")

        final_step = start_step + executed
        self.adapter.synchronize_for_observation()
        final_observation = self._save_observation(final_step)
        if self.run_spec.diagnostics:
            self._capture_diagnostic(final_step)
            write_plane_diagnostics(
                self.output_directory,
                tuple(self._diagnostics),
            )
        elapsed = time.time() - started
        integrator = self.adapter.solver.integrator
        self.metadata["completed_steps"] = final_step
        self.metadata["elapsed_seconds"] = elapsed
        self.metadata["numerics"]["spectral_refresh"]["actual_count"] = int(
            integrator.refresh_count
        )
        self.metadata["workflow"].update(
            saved_steps=list(self._saved_steps),
            checkpoint_steps=list(self._checkpoint_steps),
            final_step=final_step,
        )
        write_run_metadata(
            self.output_directory,
            self.metadata,
            status="complete",
        )
        # This is intentionally the final write in a successful workflow.
        (self.output_directory / "COMPLETE").write_text(
            "complete\n",
            encoding="utf-8",
        )
        return PlaneWorkflowResult(
            start_step=start_step,
            final_step=final_step,
            elapsed_seconds=elapsed,
            saved_steps=tuple(self._saved_steps),
            checkpoint_steps=tuple(self._checkpoint_steps),
            final_observation=final_observation,
            diagnostics=tuple(self._diagnostics),
        )


__all__ = [
    "PLANE_WORKFLOW_SCHEMA_VERSION",
    "PlaneBerisEdwardsWorkflow",
    "PlaneWorkflowResult",
]
