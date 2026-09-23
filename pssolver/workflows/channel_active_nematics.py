"""Backend-neutral finite-run workflow for the rectangular Channel."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time

from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
from pssolver.run_metadata import write_run_metadata
from pssolver.runtime.channel_active_nematics import ChannelRuntimeAdapterProtocol

from .channel_checkpoint import (
    CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    capture_channel_checkpoint, load_channel_checkpoint,
    restore_channel_checkpoint, write_channel_checkpoint,
)
from .channel_observation import (
    ChannelDiagnostic, ChannelObservation, capture_channel_diagnostic,
    capture_channel_observation, write_channel_diagnostics,
    write_channel_observation,
)


CHANNEL_WORKFLOW_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChannelWorkflowResult:
    start_step: int
    final_step: int
    elapsed_seconds: float
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_observation: ChannelObservation
    diagnostics: tuple[ChannelDiagnostic, ...]


class ChannelActiveNematicsWorkflow:
    def __init__(self, adapter: ChannelRuntimeAdapterProtocol, run_spec: ChannelActiveNematicRunSpec, output_directory: str | Path, metadata: Mapping[str, object]) -> None:
        if not isinstance(adapter, ChannelRuntimeAdapterProtocol):
            raise TypeError("adapter must implement ChannelRuntimeAdapterProtocol")
        if not isinstance(run_spec, ChannelActiveNematicRunSpec):
            raise TypeError("run_spec must be ChannelActiveNematicRunSpec")
        if adapter.runtime_path is not run_spec.runtime_path:
            raise ValueError("workflow runtime path differs from run specification")
        directory = Path(output_directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Channel output directory is missing: {directory}")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        self.adapter, self.run_spec, self.output_directory = adapter, run_spec, directory
        self.metadata = deepcopy(dict(metadata))
        self._diagnostics: list[ChannelDiagnostic] = []
        self._saved_steps: list[int] = []
        self._checkpoint_steps: list[int] = []

    @property
    def workflow(self):
        return self.run_spec.components.workflow

    def _diagnose(self, step: int) -> ChannelDiagnostic:
        material = self.run_spec.components.material
        value = capture_channel_diagnostic(
            self.adapter, step=step, beta=material.beta,
            viscosity=material.viscosity, friction=material.friction,
        )
        self._diagnostics.append(value)
        return value

    def _observe(self, step: int) -> ChannelObservation:
        value = capture_channel_observation(self.adapter, step=step)
        write_channel_observation(self.output_directory, value)
        self._saved_steps.append(step)
        return value

    def _checkpoint(self) -> None:
        value = capture_channel_checkpoint(
            self.adapter,
            runtime_identity_sha256=self.run_spec.runtime_identity_sha256(),
        )
        write_channel_checkpoint(self.output_directory / f"checkpoint_{value.completed_steps}", value)
        self._checkpoint_steps.append(value.completed_steps)

    def _restore(self) -> int:
        source = self.workflow.restart_from
        if source is None:
            if self.adapter.completed_steps != 0:
                raise ValueError("new Channel workflow requires an unadvanced runtime")
            self.metadata["restart"] = {"kind": "fresh"}
            return 0
        checkpoint = load_channel_checkpoint(source)
        start = restore_channel_checkpoint(
            self.adapter, checkpoint,
            runtime_identity_sha256=self.run_spec.runtime_identity_sha256(),
        )
        self.metadata["restart"] = {
            "kind": "same_runtime_channel_workflow_checkpoint",
            "source": str(Path(source).expanduser().resolve()),
            "source_runtime_path": checkpoint.runtime_path.value,
            "source_runtime_identity_sha256": checkpoint.runtime_identity_sha256,
            "source_completed_steps": checkpoint.completed_steps,
            "cross_runtime_adapter_used": False,
        }
        return start

    def _pre_update(self, step: int, progress: object) -> None:
        if self.workflow.diagnostics_enabled and step % self.workflow.diagnostic_interval == 0:
            value = self._diagnose(step)
            setter = getattr(progress, "set_postfix", None)
            if callable(setter):
                setter(div_max=f"{value.div_max:.2e}", schur_it=int(value.schur_iterations), schur_rel=f"{value.schur_rel_residual:.2e}")
        if step % self.workflow.save_interval == 0:
            self._observe(step)

    def run(self, *, progress: Iterable[int] | None = None) -> ChannelWorkflowResult:
        if (self.output_directory / "COMPLETE").exists():
            raise FileExistsError("Channel output directory is already complete")
        start_step = self._restore()
        self.metadata["workflow"] = {
            "schema_version": CHANNEL_WORKFLOW_SCHEMA_VERSION,
            "runtime_path": self.adapter.runtime_path.value,
            "checkpoint_format_version": CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            "cross_runtime_restart_supported": False,
            "start_step": start_step,
            "requested_additional_steps": self.workflow.steps,
            "completion_marker_order": "metadata_then_COMPLETE",
        }
        write_run_metadata(self.output_directory, self.metadata, status="running")
        iterator = range(self.workflow.steps) if progress is None else progress
        executed, started = 0, time.time()
        for local_step in iterator:
            if local_step != executed:
                raise ValueError("progress iterator must yield consecutive local steps")
            absolute = start_step + local_step
            self.adapter.advance(1, pre_update_callback=lambda _s, _n, absolute=absolute: self._pre_update(absolute, iterator))
            executed += 1
            if self.adapter.completed_steps != start_step + executed:
                raise RuntimeError("runtime completed-step clock diverged")
            interval = self.workflow.checkpoint_interval
            if interval is not None and self.adapter.completed_steps % interval == 0:
                self._checkpoint()
        if executed != self.workflow.steps:
            raise ValueError("progress iterator length differs from requested steps")
        final_step = start_step + executed
        self.adapter.synchronize_for_observation()
        final = self._observe(final_step)
        if self.workflow.diagnostics_enabled:
            self._diagnose(final_step)
            write_channel_diagnostics(self.output_directory, tuple(self._diagnostics))
        elapsed = time.time() - started
        self.metadata.update(completed_steps=final_step, elapsed_seconds=elapsed)
        self.metadata["workflow"].update(saved_steps=list(self._saved_steps), checkpoint_steps=list(self._checkpoint_steps), final_step=final_step)
        write_run_metadata(self.output_directory, self.metadata, status="complete")
        (self.output_directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
        return ChannelWorkflowResult(start_step, final_step, elapsed, tuple(self._saved_steps), tuple(self._checkpoint_steps), final, tuple(self._diagnostics))


__all__ = ["CHANNEL_WORKFLOW_SCHEMA_VERSION", "ChannelActiveNematicsWorkflow", "ChannelWorkflowResult"]
