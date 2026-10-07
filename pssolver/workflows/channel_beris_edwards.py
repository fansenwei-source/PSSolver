"""Observation and exact-restart workflow for the P8.3 Channel runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from pssolver.io.checkpoint import (
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
)
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    CheckpointIdentityLayer,
    IdentityLayerKind,
    LegacyIdentityKey,
    SourceReleaseGeneration,
    resolve_legacy_identity_schema,
)
from pssolver.configuration.channel_beris_edwards import (
    ChannelBerisEdwardsRunSpec,
)
from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime.channel_beris_edwards import (
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
    ChannelBerisEdwardsRuntimeAdapterProtocol,
)
from .diagnostic_io import write_structured_diagnostics


CHECKPOINT_VERSION = 3
LEGACY_CHECKPOINT_VERSION_V1 = 1
LEGACY_CHECKPOINT_VERSION = 2
_SUPPORTED_CHECKPOINT_VERSIONS = frozenset(
    {LEGACY_CHECKPOINT_VERSION_V1, LEGACY_CHECKPOINT_VERSION, CHECKPOINT_VERSION}
)
_STATE_COMPONENTS = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsObservation:
    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsDiagnostic:
    step: int
    divergence_max: float
    divergence_rms: float
    pressure_mean: float
    pressure_iterations: int
    pressure_relative_residual: float

    def as_tuple(self) -> tuple[int | float, ...]:
        return (
            self.step,
            self.divergence_max,
            self.divergence_rms,
            self.pressure_mean,
            self.pressure_iterations,
            self.pressure_relative_residual,
        )


CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("divergence_max", np.float64),
        ("divergence_rms", np.float64),
        ("pressure_mean", np.float64),
        ("pressure_iterations", np.int64),
        ("pressure_relative_residual", np.float64),
    ]
)
CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER = (
    "step,divergence_max,divergence_rms,pressure_mean,pressure_iterations,"
    "pressure_relative_residual"
)


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsWorkflowResult:
    start_step: int
    final_step: int
    elapsed_seconds: float
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_observation: ChannelBerisEdwardsObservation
    diagnostics: tuple[ChannelBerisEdwardsDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsCheckpointHeader:
    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    compatibility_identity: CheckpointCompatibilityIdentity | None = None

    @property
    def compatibility_identity_sha256(self) -> str | None:
        identity = self.compatibility_identity
        return None if identity is None else identity.canonical_sha256()


def capture_channel_beris_edwards_observation(adapter, *, step=None):
    if not isinstance(adapter, ChannelBerisEdwardsRuntimeAdapterProtocol):
        raise TypeError(
            "adapter must implement ChannelBerisEdwardsRuntimeAdapterProtocol"
        )
    views = adapter.output_views
    return ChannelBerisEdwardsObservation(
        adapter.completed_steps if step is None else step,
        np.moveaxis(views.q[:, 0].detach().cpu().numpy(), 0, -1),
        np.moveaxis(views.velocity[:, 0].detach().cpu().numpy(), 0, -1),
        views.pressure[0].detach().cpu().numpy(),
    )


def capture_channel_beris_edwards_diagnostic(adapter, *, step):
    fields = adapter.fields
    divergence = sum(
        adapter.projector.inverse_transform_full(
            *fields.gradient_hat(name, axis=axis)
        )
        for axis, name in enumerate(VELOCITY_COMPONENTS)
    )
    flow = adapter.flow_diagnostics()
    return ChannelBerisEdwardsDiagnostic(
        step=step,
        divergence_max=float(divergence.abs().max().item()),
        divergence_rms=float(torch.sqrt(divergence.square().mean()).item()),
        pressure_mean=float(fields["p"].mean().item()),
        pressure_iterations=flow["last_pressure_iterations"],
        pressure_relative_residual=flow["last_pressure_relative_residual"],
    )


def _save_observation(directory, observation, *, hydrodynamics):
    values = {"Q": observation.q}
    if hydrodynamics:
        values.update(u=observation.velocity, p=observation.pressure)
    for prefix, value in values.items():
        path = directory / f"{prefix}_{observation.step}.npy"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)


def _tensor_record(path, values):
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
    }


def _channel_run_spec_dynamics(
    run_spec: ChannelBerisEdwardsRunSpec,
) -> dict[str, object]:
    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a ChannelBerisEdwardsRunSpec")
    simulation = run_spec.simulation
    execution = simulation.execution.to_metadata()
    execution["options"].pop("device")
    return {
        "scientific": simulation.scientific_identity_metadata(),
        "discretization": simulation.discretization_identity_metadata(),
        "execution": execution,
    }


def channel_forward_dynamics_identity(
    run_spec: ChannelBerisEdwardsRunSpec,
) -> CheckpointIdentityLayer:
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.FORWARD_DYNAMICS,
        version="channel.production.forward.v3",
        payload={
            "run_spec_dynamics": _channel_run_spec_dynamics(run_spec),
            "dynamics_versions": {
                "q_evolution": "complete_stress_beris_edwards.v1",
                "forward_stokes_nyquist": (
                    "channel_no_slip.periodic_nyquist_zero.v2"
                ),
                "pressure_warm_start": "pcg_persistent_guess.v1",
                "spectral_refresh": "projected_euler_refresh.v1",
            },
        },
    )


def _channel_state_layout_identity(
    fields: Mapping[str, torch.Tensor],
    pressure_guess: torch.Tensor,
) -> CheckpointIdentityLayer:
    def record(value: torch.Tensor) -> dict[str, object]:
        return {"shape": list(value.shape), "dtype": str(value.dtype)}

    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.STATE_LAYOUT,
        version="channel.production.state.v3",
        payload={
            "component_order": list(_STATE_COMPONENTS),
            "spatial": {
                name: record(fields[name]) for name in _STATE_COMPONENTS
            },
            "spectral": {
                name: record(fields[f"{name}.hat"])
                for name in _STATE_COMPONENTS
            },
            "persistent": {"pressure_guess": record(pressure_guess)},
        },
    )


def _channel_backend_restart_identity(
    backend_restart: dict[str, object],
) -> CheckpointIdentityLayer:
    normalized = json.loads(
        json.dumps(
            dict(backend_restart),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    normalized.pop("runtime_identity_sha256", None)
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.BACKEND_RESTART,
        version="channel.production.backend.v1",
        payload=normalized,
    )


def build_channel_checkpoint_compatibility_identity(
    run_spec: ChannelBerisEdwardsRunSpec,
    *,
    fields: Mapping[str, torch.Tensor],
    pressure_guess: torch.Tensor,
    backend_restart: dict[str, object],
) -> CheckpointCompatibilityIdentity:
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.CHANNEL_WORKFLOW,
        runtime_path=CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
        forward_dynamics=channel_forward_dynamics_identity(run_spec),
        state_layout=_channel_state_layout_identity(fields, pressure_guess),
        backend_restart=_channel_backend_restart_identity(backend_restart),
    )


def write_channel_beris_edwards_checkpoint(
    directory,
    adapter,
    *,
    run_spec,
):
    """Write the complete-stress Channel production checkpoint format."""

    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: {directory}")
    directory.mkdir(parents=True)
    adapter.synchronize_for_observation()
    fields = adapter.fields
    records = {"spatial": {}, "spectral": {}}
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        for name in _STATE_COMPONENTS:
            values = fields[f"{name}{suffix}"].detach().cpu().numpy()
            path = directory / f"{kind}__{name}.npy"
            records[kind][name] = _tensor_record(path, values)
    pressure_guess = adapter.capture_pressure_guess().cpu().numpy()
    pressure_record = _tensor_record(
        directory / "backend__pressure_guess.npy",
        pressure_guess,
    )
    integrator = adapter.solver.integrator
    backend_restart = adapter.backend_restart_metadata()
    compatibility_identity = build_channel_checkpoint_compatibility_identity(
        run_spec,
        fields=fields,
        pressure_guess=adapter.capture_pressure_guess(),
        backend_restart=backend_restart,
    )
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "runtime_path": CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
        "run_provenance": {
            "legacy_runtime_identity_sha256": (
                run_spec.runtime_identity_sha256()
            ),
        },
        "compatibility_identity": compatibility_identity.to_metadata(),
        "compatibility_identity_sha256": (
            compatibility_identity.canonical_sha256()
        ),
        "completed_steps": adapter.completed_steps,
        "integrator": {
            "spectral_refresh_interval": integrator.spectral_refresh_interval,
            "step_count": int(integrator.step_count),
            "refresh_count": int(integrator.refresh_count),
        },
        "backend_restart": backend_restart,
        "tensor_files": records,
        "backend_files": {"pressure_guess": pressure_record},
    }
    metadata = seal_checkpoint_metadata(metadata)
    (directory / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validated_array(directory, record, target, description):
    if set(record) != {"file", "shape", "dtype", "sha256"}:
        raise ValueError(f"{description} record schema differs")
    path = directory / record["file"]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{description} checksum mismatch")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
        raise ValueError(f"{description} metadata differs")
    if tuple(values.shape) != tuple(target.shape):
        raise ValueError(f"{description} shape differs")
    if str(values.dtype) != str(target.detach().cpu().numpy().dtype):
        raise ValueError(f"{description} dtype differs")
    if not np.isfinite(values).all():
        raise ValueError(f"{description} is not finite")
    return values


def _validated_progress(metadata):
    completed_steps = metadata.get("completed_steps")
    if (
        not isinstance(completed_steps, int)
        or isinstance(completed_steps, bool)
        or completed_steps < 0
    ):
        raise ValueError("checkpoint completed_steps is invalid")
    integrator = metadata.get("integrator")
    if not isinstance(integrator, dict) or set(integrator) != {
        "spectral_refresh_interval",
        "step_count",
        "refresh_count",
    }:
        raise ValueError("checkpoint integrator metadata differs")
    interval = integrator["spectral_refresh_interval"]
    if interval is not None and (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError("checkpoint spectral refresh interval is invalid")
    for name in ("step_count", "refresh_count"):
        value = integrator[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"checkpoint {name} is invalid")
    if interval is None:
        expected_refresh_count = 0
        expected_step_count = completed_steps
    else:
        expected_refresh_count, expected_step_count = divmod(
            completed_steps,
            interval,
        )
    if (
        integrator["step_count"] != expected_step_count
        or integrator["refresh_count"] != expected_refresh_count
    ):
        raise ValueError("checkpoint progress counters differ")
    return completed_steps, integrator


def _read_channel_checkpoint_metadata(directory: Path) -> dict[str, object]:
    path = directory / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a JSON object")
    format_version = metadata.get("format_version")
    if (
        not isinstance(format_version, int)
        or isinstance(format_version, bool)
        or format_version not in _SUPPORTED_CHECKPOINT_VERSIONS
    ):
        raise ValueError("unsupported complete-stress Channel checkpoint")
    verify_checkpoint_metadata(
        metadata,
        required=format_version in {LEGACY_CHECKPOINT_VERSION, CHECKPOINT_VERSION},
    )
    if metadata.get("runtime_path") != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
        raise ValueError("checkpoint runtime identity does not match target")
    return metadata


def read_channel_beris_edwards_checkpoint_header(
    directory,
) -> ChannelBerisEdwardsCheckpointHeader:
    directory = Path(directory).expanduser().resolve()
    metadata = _read_channel_checkpoint_metadata(directory)
    format_version = metadata.get("format_version")
    completed_steps, _ = _validated_progress(metadata)
    compatibility_identity = None
    if format_version == CHECKPOINT_VERSION:
        run_provenance = metadata.get("run_provenance")
        if not isinstance(run_provenance, Mapping):
            raise ValueError("Channel checkpoint run provenance is missing")
        runtime_identity = run_provenance.get("legacy_runtime_identity_sha256")
        compatibility_identity = CheckpointCompatibilityIdentity.from_metadata(
            metadata.get("compatibility_identity")
        )
        expected = metadata.get("compatibility_identity_sha256")
        if compatibility_identity.canonical_sha256() != expected:
            raise ValueError("Channel compatibility identity checksum mismatch")
        if compatibility_identity.family is not CheckpointFamily.CHANNEL_WORKFLOW:
            raise ValueError("Channel compatibility identity family differs")
        if (
            compatibility_identity.runtime_path
            != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH
        ):
            raise ValueError("Channel compatibility runtime path differs")
    else:
        runtime_identity = metadata.get("runtime_identity_sha256")
    if (
        not isinstance(runtime_identity, str)
        or len(runtime_identity) != 64
        or any(value not in "0123456789abcdef" for value in runtime_identity)
    ):
        raise ValueError("Channel runtime identity must be a lowercase SHA-256")
    return ChannelBerisEdwardsCheckpointHeader(
        format_version=format_version,
        runtime_identity_sha256=runtime_identity,
        completed_steps=completed_steps,
        compatibility_identity=compatibility_identity,
    )


def read_channel_beris_edwards_checkpoint_progress(directory):
    """Read current authenticated progress without opening checkpoint tensors."""

    header = read_channel_beris_edwards_checkpoint_header(directory)
    if header.compatibility_identity is None:
        raise ValueError("legacy Channel checkpoint requires explicit identity upgrade")
    return header.completed_steps


def _load_checkpoint(directory, adapter, *, run_spec):
    directory = Path(directory).expanduser().resolve()
    metadata = _read_channel_checkpoint_metadata(directory)
    header = read_channel_beris_edwards_checkpoint_header(directory)
    if header.compatibility_identity is None:
        raise ValueError("legacy Channel checkpoint requires explicit identity upgrade")
    completed_steps, integrator = _validated_progress(metadata)
    if header.compatibility_identity.state_layout != (
        _channel_state_layout_identity_from_metadata(metadata)
    ):
        raise ValueError("Channel state layout identity is inconsistent")
    backend_metadata = metadata.get("backend_restart")
    if not isinstance(backend_metadata, dict) or (
        header.compatibility_identity.backend_restart
        != _channel_backend_restart_identity(backend_metadata)
    ):
        raise ValueError("Channel backend restart identity is inconsistent")

    fields = adapter.fields
    pressure_guess = adapter.capture_pressure_guess()
    expected_identity = build_channel_checkpoint_compatibility_identity(
        run_spec,
        fields=fields,
        pressure_guess=pressure_guess,
        backend_restart=adapter.backend_restart_metadata(),
    )
    if header.compatibility_identity != expected_identity:
        raise ValueError("Channel compatibility identity does not match target")
    pending = []
    manifest = metadata.get("tensor_files", {})
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        records = manifest.get(kind, {})
        if set(records) != set(_STATE_COMPONENTS):
            raise ValueError("Channel checkpoint tensor manifest is incomplete")
        for name in _STATE_COMPONENTS:
            target = fields[f"{name}{suffix}"]
            values = _validated_array(
                directory,
                records[name],
                target,
                f"{kind} {name}",
            )
            pending.append((target, values))
    pressure_target = fields["p.hat"]
    pressure_record = metadata.get("backend_files", {}).get("pressure_guess")
    pressure_values = _validated_array(
        directory,
        pressure_record,
        pressure_target,
        "pressure guess",
    )

    # Mutation starts only after every identity, file, shape, dtype, checksum,
    # and finite-value gate has passed.
    for target, values in pending:
        target.copy_(torch.from_numpy(values).to(device=target.device))
    adapter.restore_pressure_guess(
        torch.from_numpy(pressure_values).to(device=pressure_target.device)
    )
    adapter.restore_progress(
        completed_steps=completed_steps,
        spectral_refresh_interval=integrator["spectral_refresh_interval"],
        integrator_step_count=integrator["step_count"],
        integrator_refresh_count=integrator["refresh_count"],
    )
    adapter.restore_derived_state()
    return completed_steps


def _channel_state_layout_identity_from_metadata(
    metadata: Mapping[str, object],
) -> CheckpointIdentityLayer:
    manifest = metadata.get("tensor_files")
    backend_files = metadata.get("backend_files")
    if not isinstance(manifest, Mapping) or not isinstance(
        backend_files, Mapping
    ):
        raise ValueError("Channel checkpoint tensor manifest is missing")

    def layout(record: object, description: str) -> dict[str, object]:
        if not isinstance(record, Mapping):
            raise ValueError(f"{description} tensor record is missing")
        shape = record.get("shape")
        dtype = record.get("dtype")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in shape
            )
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"{description} tensor layout is invalid")
        return {"shape": shape, "dtype": f"torch.{dtype}"}

    result: dict[str, object] = {
        "component_order": list(_STATE_COMPONENTS),
        "spatial": {},
        "spectral": {},
        "persistent": {},
    }
    for kind in ("spatial", "spectral"):
        records = manifest.get(kind)
        if not isinstance(records, Mapping) or set(records) != set(
            _STATE_COMPONENTS
        ):
            raise ValueError("Channel checkpoint tensor manifest is incomplete")
        result[kind] = {
            name: layout(records[name], f"{kind} {name}")
            for name in _STATE_COMPONENTS
        }
    if set(backend_files) != {"pressure_guess"}:
        raise ValueError("Channel checkpoint backend manifest is incomplete")
    result["persistent"] = {
        "pressure_guess": layout(
            backend_files["pressure_guess"],
            "pressure guess",
        )
    }
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.STATE_LAYOUT,
        version="channel.production.state.v3",
        payload=result,
    )


def _legacy_channel_format_for_generation(
    generation: SourceReleaseGeneration,
) -> int:
    if not isinstance(generation, SourceReleaseGeneration):
        raise TypeError("source_release_generation must be a SourceReleaseGeneration")
    return (
        LEGACY_CHECKPOINT_VERSION_V1
        if generation is SourceReleaseGeneration.RC1
        else LEGACY_CHECKPOINT_VERSION
    )


def _validate_legacy_channel_upgrade_contract(
    metadata: Mapping[str, object],
    *,
    source_run_spec: ChannelBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    if not isinstance(source_run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("source_run_spec must be a ChannelBerisEdwardsRunSpec")
    expected_format = _legacy_channel_format_for_generation(
        source_release_generation
    )
    if metadata.get("format_version") != expected_format:
        raise ValueError("legacy Channel format does not match source release")
    if (
        metadata.get("runtime_identity_sha256")
        != source_run_spec.runtime_identity_sha256()
    ):
        raise ValueError(
            "legacy Channel checkpoint opaque identity is not authenticated"
        )
    # RC4.2.1 froze the Channel identity schema as v1 for every release;
    # the transport container independently became sealed format v2 in rc2.
    schema = resolve_legacy_identity_schema(
        LegacyIdentityKey(
            checkpoint_family=CheckpointFamily.CHANNEL_WORKFLOW,
            format_version=LEGACY_CHECKPOINT_VERSION_V1,
            source_release_generation=source_release_generation,
            runtime_path=CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
            applicability_class="configuration_dependent",
        )
    )
    periodic_length = source_run_spec.simulation.geometry.domain.shape[0]
    if (
        source_release_generation is SourceReleaseGeneration.RC1
        and periodic_length % 2 == 0
    ):
        raise ValueError(
            "rc1 Channel checkpoint has pre-RC4.1 forward Nyquist dynamics"
        )
    return schema, periodic_length


def upgrade_channel_beris_edwards_checkpoint_directory(
    source_directory,
    target_directory,
    *,
    source_run_spec: ChannelBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    """Write a layered v3 copy after legacy forward-dynamics adjudication."""

    source = Path(source_directory).expanduser().resolve()
    target = Path(target_directory).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("legacy upgrade target must be outside the source checkpoint")
    metadata = _read_channel_checkpoint_metadata(source)
    if metadata.get("format_version") == CHECKPOINT_VERSION:
        raise ValueError("current Channel checkpoint does not require upgrade")
    schema, periodic_length = _validate_legacy_channel_upgrade_contract(
        metadata,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    completed_steps, _ = _validated_progress(metadata)
    state_layout = _channel_state_layout_identity_from_metadata(metadata)
    backend = metadata.get("backend_restart")
    if not isinstance(backend, dict):
        raise ValueError("Channel backend restart metadata is missing")
    compatibility_identity = CheckpointCompatibilityIdentity(
        family=CheckpointFamily.CHANNEL_WORKFLOW,
        runtime_path=CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
        forward_dynamics=channel_forward_dynamics_identity(source_run_spec),
        state_layout=state_layout,
        backend_restart=_channel_backend_restart_identity(backend),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        upgraded = dict(metadata)
        upgraded.pop("metadata_sha256", None)
        upgraded.pop("runtime_identity_sha256", None)
        upgraded["format_version"] = CHECKPOINT_VERSION
        upgraded["run_provenance"] = {
            "legacy_runtime_identity_sha256": source_run_spec.runtime_identity_sha256()
        }
        upgraded["compatibility_identity"] = compatibility_identity.to_metadata()
        upgraded["compatibility_identity_sha256"] = (
            compatibility_identity.canonical_sha256()
        )
        upgraded["migration_provenance"] = {
            "kind": "channel_checkpoint_identity_upgrade_to_v3",
            "source_release_generation": source_release_generation.value,
            "source_commit": schema.source_commit,
            "source_format_version": metadata["format_version"],
            "legacy_schema_id": schema.schema_id,
            "legacy_canonicalizer_id": schema.canonicalizer_id,
            "periodic_axis_length": periodic_length,
            "forward_nyquist_applicability": (
                "inapplicable_odd_periodic_axis"
                if source_release_generation is SourceReleaseGeneration.RC1
                else "source_is_post_nyquist_repair"
            ),
            "source_completed_steps": completed_steps,
            "source_checkpoint_mutated": False,
        }
        upgraded = seal_checkpoint_metadata(upgraded)
        (staging / "checkpoint.json").write_text(
            json.dumps(upgraded, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


class ChannelBerisEdwardsWorkflow:
    def __init__(self, adapter, run_spec, output_directory, metadata):
        if not isinstance(adapter, ChannelBerisEdwardsRuntimeAdapterProtocol):
            raise TypeError(
                "adapter must implement "
                "ChannelBerisEdwardsRuntimeAdapterProtocol"
            )
        self.adapter = adapter
        self.run_spec = run_spec
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.metadata = dict(metadata)
        self.options = dict(run_spec.simulation.workflow.options)
        self.saved = []
        self.checkpoints = []
        self.diagnostics = []

    def run(self, *, progress=None):
        restart = self.options["restart_from"]
        start = 0 if restart is None else _load_checkpoint(
            restart,
            self.adapter,
            run_spec=self.run_spec,
        )
        steps = self.run_spec.simulation.workflow.steps
        if self.options["save_start_step"] > start + steps:
            raise ValueError(
                "save_start_step must not exceed the absolute final step"
            )
        iterator = range(steps) if progress is None else progress
        started = time.time()
        executed = 0
        for local_step in iterator:
            if local_step != executed:
                raise ValueError(
                    "progress must yield consecutive local steps from zero"
                )
            absolute = start + local_step
            diagnostic_due = (
                self.options["diagnostics"]
                and absolute % self.options["diagnostic_interval"] == 0
            )
            save_due = (
                absolute >= self.options["save_start_step"]
                and absolute % self.options["save_interval"] == 0
            )
            if diagnostic_due or save_due:
                self.adapter.synchronize_for_observation()
            if diagnostic_due:
                self.diagnostics.append(
                    capture_channel_beris_edwards_diagnostic(
                        self.adapter,
                        step=absolute,
                    )
                )
            if save_due:
                observation = capture_channel_beris_edwards_observation(
                    self.adapter,
                    step=absolute,
                )
                _save_observation(
                    self.output_directory,
                    observation,
                    hydrodynamics=self.options["save_hydrodynamics"],
                )
                self.saved.append(absolute)
            self.adapter.advance(1)
            executed += 1
            interval = self.options["checkpoint_interval"]
            if interval is not None and self.adapter.completed_steps % interval == 0:
                write_channel_beris_edwards_checkpoint(
                    self.output_directory
                    / f"checkpoint_{self.adapter.completed_steps}",
                    self.adapter,
                    run_spec=self.run_spec,
                )
                self.checkpoints.append(self.adapter.completed_steps)
        if executed != steps:
            raise ValueError(
                f"progress yielded {executed} steps; expected {steps}"
            )
        final_step = start + steps
        self.adapter.synchronize_for_observation()
        final = capture_channel_beris_edwards_observation(
            self.adapter,
            step=final_step,
        )
        if final_step not in self.saved:
            _save_observation(
                self.output_directory,
                final,
                hydrodynamics=self.options["save_hydrodynamics"],
            )
            self.saved.append(final_step)
        if self.options["diagnostics"]:
            self.diagnostics.append(
                capture_channel_beris_edwards_diagnostic(
                    self.adapter,
                    step=final_step,
                )
            )
            write_structured_diagnostics(
                self.output_directory,
                [value.as_tuple() for value in self.diagnostics],
                dtype=CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE,
                header=CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER,
            )
        elapsed = time.time() - started
        self.metadata.update(
            status="complete",
            completed_steps=final_step,
            elapsed_seconds=elapsed,
            saved_steps=self.saved,
            checkpoint_steps=self.checkpoints,
            diagnostics={
                "enabled": self.options["diagnostics"],
                "count": len(self.diagnostics),
                "steps": [value.step for value in self.diagnostics],
                "npy": (
                    "diagnostics.npy" if self.options["diagnostics"] else None
                ),
                "csv": (
                    "diagnostics.csv" if self.options["diagnostics"] else None
                ),
            },
        )
        (self.output_directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_directory / "COMPLETE").write_text(
            "complete\n",
            encoding="utf-8",
        )
        return ChannelBerisEdwardsWorkflowResult(
            start,
            final_step,
            elapsed,
            tuple(self.saved),
            tuple(self.checkpoints),
            final,
            tuple(self.diagnostics),
        )


__all__ = [
    "CHECKPOINT_VERSION",
    "LEGACY_CHECKPOINT_VERSION",
    "LEGACY_CHECKPOINT_VERSION_V1",
    "ChannelBerisEdwardsDiagnostic",
    "CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE",
    "CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER",
    "ChannelBerisEdwardsObservation",
    "ChannelBerisEdwardsCheckpointHeader",
    "ChannelBerisEdwardsWorkflow",
    "ChannelBerisEdwardsWorkflowResult",
    "build_channel_checkpoint_compatibility_identity",
    "capture_channel_beris_edwards_diagnostic",
    "capture_channel_beris_edwards_observation",
    "channel_forward_dynamics_identity",
    "read_channel_beris_edwards_checkpoint_header",
    "read_channel_beris_edwards_checkpoint_progress",
    "upgrade_channel_beris_edwards_checkpoint_directory",
    "write_channel_beris_edwards_checkpoint",
]
