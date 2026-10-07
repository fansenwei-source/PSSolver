"""Versioned exact-restart checkpoints for the periodic Beris--Edwards runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

import numpy as np
import torch

from pssolver.configuration.periodic_beris_edwards import (
    PERIODIC_RUNTIME_PATH,
    PeriodicBerisEdwardsRunSpec,
)
from pssolver.io.checkpoint import (
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
    verify_functional_checkpoint_provenance,
    write_functional_checkpoint_provenance,
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
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import PeriodicRuntimeAdapterProtocol


PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1 = 1
PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 2
PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 3
_SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS = frozenset(
    {
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1,
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _json_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be JSON-compatible") from exc
    return MappingProxyType(json.loads(encoded))


def _clone_tensors(
    values: Mapping[str, torch.Tensor],
    description: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(values, Mapping) or tuple(values) != Q_COMPONENTS:
        raise ValueError(f"{description} must contain ordered Q components")
    cloned = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} values must be tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{description} tensors must be finite")
        cloned[name] = value.detach().clone()
    return MappingProxyType(cloned)


def _validate_progress(
    completed_steps: int,
    spectral_refresh_interval: int | None,
    integrator_step_count: int,
    integrator_refresh_count: int,
) -> None:
    for value, description in (
        (completed_steps, "completed_steps"),
        (integrator_step_count, "integrator_step_count"),
        (integrator_refresh_count, "integrator_refresh_count"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{description} must be non-negative")
    interval = spectral_refresh_interval
    if interval is not None and (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError("spectral refresh interval must be positive or None")
    if interval is None:
        valid = integrator_refresh_count == 0 and integrator_step_count == completed_steps
    else:
        valid = (
            integrator_step_count < interval
            and integrator_refresh_count * interval + integrator_step_count
            == completed_steps
        )
    if not valid:
        raise ValueError("checkpoint refresh counters are inconsistent")


@dataclass(frozen=True, slots=True)
class PeriodicCheckpointHeader:
    """Small identity record read before any checkpoint array is loaded."""

    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    backend_restart: Mapping[str, object]
    functional_bridge: Mapping[str, object] | None = None
    compatibility_identity: CheckpointCompatibilityIdentity | None = None

    @property
    def compatibility_identity_sha256(self) -> str | None:
        identity = self.compatibility_identity
        return None if identity is None else identity.canonical_sha256()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version
            not in _SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS
        ):
            raise ValueError("unsupported periodic workflow checkpoint version")
        _require_sha256(self.runtime_identity_sha256, "runtime_identity_sha256")
        _validate_progress(
            self.completed_steps,
            self.spectral_refresh_interval,
            self.integrator_step_count,
            self.integrator_refresh_count,
        )
        object.__setattr__(
            self,
            "backend_restart",
            _json_mapping(self.backend_restart, "backend_restart"),
        )
        if self.functional_bridge is not None:
            object.__setattr__(
                self,
                "functional_bridge",
                _json_mapping(self.functional_bridge, "functional_bridge"),
            )
        identity = self.compatibility_identity
        if self.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            if not isinstance(identity, CheckpointCompatibilityIdentity):
                raise TypeError("periodic compatibility identity is missing")
            if identity.family is not CheckpointFamily.PERIODIC_WORKFLOW:
                raise ValueError("periodic compatibility identity family differs")
            if identity.runtime_path != PERIODIC_RUNTIME_PATH:
                raise ValueError("periodic compatibility runtime path differs")
        elif identity is not None:
            raise ValueError("legacy periodic checkpoints cannot carry v3 identity")


@dataclass(frozen=True, slots=True)
class PeriodicWorkflowCheckpoint:
    """Complete evolved Q state and integrator phase for periodic execution."""

    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    evolved_spatial: Mapping[str, torch.Tensor]
    evolved_spectral: Mapping[str, torch.Tensor]
    backend_restart: Mapping[str, object]
    functional_bridge: Mapping[str, object] | None = None
    compatibility_identity: CheckpointCompatibilityIdentity | None = None
    migration_provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version
            not in _SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS
        ):
            raise ValueError("unsupported periodic workflow checkpoint version")
        _require_sha256(self.runtime_identity_sha256, "runtime_identity_sha256")
        _validate_progress(
            self.completed_steps,
            self.spectral_refresh_interval,
            self.integrator_step_count,
            self.integrator_refresh_count,
        )
        object.__setattr__(
            self,
            "evolved_spatial",
            _clone_tensors(self.evolved_spatial, "evolved_spatial"),
        )
        object.__setattr__(
            self,
            "evolved_spectral",
            _clone_tensors(self.evolved_spectral, "evolved_spectral"),
        )
        object.__setattr__(
            self,
            "backend_restart",
            _json_mapping(self.backend_restart, "backend_restart"),
        )
        if self.functional_bridge is not None:
            object.__setattr__(
                self,
                "functional_bridge",
                _json_mapping(self.functional_bridge, "functional_bridge"),
            )
        identity = self.compatibility_identity
        if self.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            if not isinstance(identity, CheckpointCompatibilityIdentity):
                raise TypeError("periodic compatibility identity is missing")
            if identity.family is not CheckpointFamily.PERIODIC_WORKFLOW:
                raise ValueError("periodic compatibility identity family differs")
            if identity.runtime_path != PERIODIC_RUNTIME_PATH:
                raise ValueError("periodic compatibility runtime path differs")
        elif identity is not None:
            raise ValueError("legacy periodic checkpoints cannot carry v3 identity")
        if self.migration_provenance is not None:
            object.__setattr__(
                self,
                "migration_provenance",
                _json_mapping(self.migration_provenance, "migration_provenance"),
            )

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format_version": self.format_version,
            "runtime_path": PERIODIC_RUNTIME_PATH,
            "completed_steps": self.completed_steps,
            "integrator": {
                "spectral_refresh_interval": self.spectral_refresh_interval,
                "step_count": self.integrator_step_count,
                "refresh_count": self.integrator_refresh_count,
            },
            "backend_restart": dict(self.backend_restart),
        }
        if self.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            identity = self.compatibility_identity
            if identity is None:
                raise RuntimeError("periodic compatibility identity is missing")
            metadata["run_provenance"] = {
                "legacy_runtime_identity_sha256": self.runtime_identity_sha256,
            }
            metadata["compatibility_identity"] = identity.to_metadata()
            metadata["compatibility_identity_sha256"] = (
                identity.canonical_sha256()
            )
        else:
            metadata["runtime_identity_sha256"] = self.runtime_identity_sha256
        if self.functional_bridge is not None:
            metadata["functional_bridge"] = dict(self.functional_bridge)
        if self.migration_provenance is not None:
            metadata["migration_provenance"] = dict(self.migration_provenance)
        return metadata


def _periodic_run_spec_dynamics(
    run_spec: PeriodicBerisEdwardsRunSpec,
) -> dict[str, object]:
    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PeriodicBerisEdwardsRunSpec")
    simulation = run_spec.simulation
    execution = simulation.execution.to_metadata()
    execution["options"].pop("device")
    return {
        "scientific": simulation.scientific_identity_metadata(),
        "discretization": simulation.discretization_identity_metadata(),
        "execution": execution,
    }


def periodic_forward_dynamics_identity(
    run_spec: PeriodicBerisEdwardsRunSpec,
) -> CheckpointIdentityLayer:
    """Return periodic next-step semantics without allocation provenance."""

    storage = run_spec.simulation.numerics.spectral_storage.value
    hermitian_projection = {
        "full_complex": "complete_conjugate_pair_projection_each_step.v2",
        "hermitian_half": "self_conjugate_stored_planes_projection_each_step.v2",
    }[storage]
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.FORWARD_DYNAMICS,
        version="periodic.forward.v3",
        payload={
            "run_spec_dynamics": _periodic_run_spec_dynamics(run_spec),
            "dynamics_versions": {
                "q_evolution": "complete_stress_beris_edwards.v1",
                "hermitian_projection": hermitian_projection,
                "stokes_nyquist": "periodic_modal_stokes.nyquist_zero.v1",
                "spectral_refresh": "projected_euler_refresh.v1",
            },
        },
    )


def _periodic_state_layout_identity(
    evolved_spatial: Mapping[str, torch.Tensor],
    evolved_spectral: Mapping[str, torch.Tensor],
) -> CheckpointIdentityLayer:
    def layout(values: Mapping[str, torch.Tensor]) -> dict[str, object]:
        return {
            name: {
                "shape": list(values[name].shape),
                "dtype": str(values[name].dtype),
            }
            for name in Q_COMPONENTS
        }

    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.STATE_LAYOUT,
        version="periodic.production.state.v3",
        payload={
            "component_order": list(Q_COMPONENTS),
            "spatial": layout(evolved_spatial),
            "spectral": layout(evolved_spectral),
        },
    )


def _periodic_backend_restart_identity(
    backend_restart: Mapping[str, object],
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
        version="periodic.production.backend.v1",
        payload=normalized,
    )


def build_periodic_checkpoint_compatibility_identity(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    evolved_spatial: Mapping[str, torch.Tensor],
    evolved_spectral: Mapping[str, torch.Tensor],
    backend_restart: Mapping[str, object],
) -> CheckpointCompatibilityIdentity:
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.PERIODIC_WORKFLOW,
        runtime_path=PERIODIC_RUNTIME_PATH,
        forward_dynamics=periodic_forward_dynamics_identity(run_spec),
        state_layout=_periodic_state_layout_identity(
            evolved_spatial,
            evolved_spectral,
        ),
        backend_restart=_periodic_backend_restart_identity(backend_restart),
    )


def _validate_periodic_embedded_identity(
    checkpoint: PeriodicWorkflowCheckpoint,
) -> None:
    identity = checkpoint.compatibility_identity
    if identity is None:
        raise ValueError("periodic compatibility identity is missing")
    if identity.state_layout != _periodic_state_layout_identity(
        checkpoint.evolved_spatial,
        checkpoint.evolved_spectral,
    ):
        raise ValueError("periodic state layout identity is inconsistent")
    if identity.backend_restart != _periodic_backend_restart_identity(
        checkpoint.backend_restart
    ):
        raise ValueError("periodic backend restart identity is inconsistent")


def capture_periodic_checkpoint(
    adapter: PeriodicRuntimeAdapterProtocol,
    *,
    run_spec: PeriodicBerisEdwardsRunSpec,
) -> PeriodicWorkflowCheckpoint:
    """Synchronize algebraic fields and capture the exact production state."""

    if not isinstance(adapter, PeriodicRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PeriodicRuntimeAdapterProtocol")
    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PeriodicBerisEdwardsRunSpec")
    adapter.synchronize_for_observation()
    fields = adapter.fields
    integrator = adapter.solver.integrator
    spatial = {name: fields[name] for name in Q_COMPONENTS}
    spectral = {name: fields[f"{name}.hat"] for name in Q_COMPONENTS}
    backend = adapter.backend_restart_metadata()
    return PeriodicWorkflowCheckpoint(
        format_version=PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=run_spec.runtime_identity_sha256(),
        completed_steps=adapter.completed_steps,
        spectral_refresh_interval=integrator.spectral_refresh_interval,
        integrator_step_count=int(integrator.step_count),
        integrator_refresh_count=int(integrator.refresh_count),
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        compatibility_identity=build_periodic_checkpoint_compatibility_identity(
            run_spec,
            evolved_spatial=spatial,
            evolved_spectral=spectral,
            backend_restart=backend,
        ),
    )


def restore_periodic_checkpoint(
    adapter: PeriodicRuntimeAdapterProtocol,
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    run_spec: PeriodicBerisEdwardsRunSpec,
) -> int:
    """Restore only after every identity, shape, and dtype gate has passed."""

    if not isinstance(adapter, PeriodicRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PeriodicRuntimeAdapterProtocol")
    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be PeriodicWorkflowCheckpoint")
    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PeriodicBerisEdwardsRunSpec")
    fields = adapter.fields
    if checkpoint.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        _validate_periodic_embedded_identity(checkpoint)
        target_spatial = {name: fields[name] for name in Q_COMPONENTS}
        target_spectral = {
            name: fields[f"{name}.hat"] for name in Q_COMPONENTS
        }
        target_identity = build_periodic_checkpoint_compatibility_identity(
            run_spec,
            evolved_spatial=target_spatial,
            evolved_spectral=target_spectral,
            backend_restart=adapter.backend_restart_metadata(),
        )
        if checkpoint.compatibility_identity != target_identity:
            raise ValueError(
                "periodic checkpoint compatibility identity does not match target"
            )
    elif (
        checkpoint.format_version
        == PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
        and checkpoint.functional_bridge is not None
        and checkpoint.functional_bridge.get("format_version") == 2
    ):
        # Preserve the already-qualified API-1.0 functional bridge until its
        # derivative-aware RC4.2.4 carrier is introduced.
        if (
            checkpoint.runtime_identity_sha256
            != run_spec.runtime_identity_sha256()
        ):
            raise ValueError("checkpoint scientific/runtime identity does not match")
        if dict(checkpoint.backend_restart) != adapter.backend_restart_metadata():
            raise ValueError("checkpoint backend contract does not match target")
    else:
        raise ValueError(
            "legacy periodic checkpoint requires explicit identity upgrade"
        )
    targets: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint tensor shape differs for {name}{suffix}")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint tensor dtype differs for {name}{suffix}")
            targets.append((target, source, f"{name}{suffix}"))

    # No target is mutated until the complete preflight above succeeds.
    for target, source, _description in targets:
        target.copy_(source.to(device=target.device))
    adapter.synchronize_for_observation()
    adapter.restore_progress(
        completed_steps=checkpoint.completed_steps,
        spectral_refresh_interval=checkpoint.spectral_refresh_interval,
        integrator_step_count=checkpoint.integrator_step_count,
        integrator_refresh_count=checkpoint.integrator_refresh_count,
    )
    if adapter.completed_steps != checkpoint.completed_steps:
        raise RuntimeError("restored completed-step count is inconsistent")
    return checkpoint.completed_steps


def _write_tensor(path: Path, value: torch.Tensor) -> dict[str, object]:
    array = value.detach().to(device="cpu").contiguous().numpy()
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha256(path),
    }


def write_periodic_checkpoint(
    directory: str | Path,
    checkpoint: PeriodicWorkflowCheckpoint,
) -> Path:
    """Atomically write a production-compatible checkpoint without pickle."""

    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be PeriodicWorkflowCheckpoint")
    target = Path(directory).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    # mkdtemp deliberately creates mode 0700.  The atomic rename would make
    # that private staging mode the durable checkpoint mode, preventing
    # read-only service accounts from consuming an otherwise shared run.
    staging.chmod(0o755)
    try:
        files = {"spatial": {}, "spectral": {}}
        for kind, tensors in (
            ("spatial", checkpoint.evolved_spatial),
            ("spectral", checkpoint.evolved_spectral),
        ):
            for name, value in tensors.items():
                files[kind][name] = _write_tensor(
                    staging / f"{kind}__{name}.npy",
                    value,
                )
        metadata = checkpoint.to_metadata()
        metadata["tensor_files"] = files
        if checkpoint.format_version in {
            PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        }:
            metadata = seal_checkpoint_metadata(metadata)
        if checkpoint.functional_bridge is not None:
            write_functional_checkpoint_provenance(
                staging,
                metadata,
                checkpoint.functional_bridge,
            )
        (staging / "checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _read_metadata(directory: Path) -> Mapping[str, object]:
    path = directory / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a JSON object")
    version = metadata.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS
    ):
        raise ValueError("unsupported periodic workflow checkpoint version")
    verify_checkpoint_metadata(
        metadata,
        required=version
        in {
            PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        },
    )
    bridge = metadata.get("functional_bridge")
    bridge_version = (
        bridge.get("format_version") if isinstance(bridge, Mapping) else None
    )
    verify_functional_checkpoint_provenance(
        directory,
        metadata,
        bridge,
        required=bridge_version == 2,
    )
    return metadata


def read_periodic_checkpoint_header(
    directory: str | Path,
) -> PeriodicCheckpointHeader:
    """Read all small restart identities without opening tensor payloads."""

    metadata = _read_metadata(Path(directory).expanduser().resolve())
    if metadata.get("runtime_path") != PERIODIC_RUNTIME_PATH:
        raise ValueError("checkpoint runtime identity does not match target")
    format_version = metadata.get("format_version")
    compatibility_identity = None
    if format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        run_provenance = metadata.get("run_provenance")
        if not isinstance(run_provenance, Mapping):
            raise ValueError("periodic checkpoint run provenance is missing")
        identity = _require_sha256(
            run_provenance.get("legacy_runtime_identity_sha256"),
            "legacy_runtime_identity_sha256",
        )
        compatibility_identity = (
            CheckpointCompatibilityIdentity.from_metadata(
                metadata.get("compatibility_identity")
            )
        )
        expected = _require_sha256(
            metadata.get("compatibility_identity_sha256"),
            "compatibility_identity_sha256",
        )
        if compatibility_identity.canonical_sha256() != expected:
            raise ValueError(
                "periodic checkpoint compatibility identity checksum mismatch"
            )
    else:
        identity = _require_sha256(
            metadata.get("runtime_identity_sha256"),
            "runtime_identity_sha256",
        )
    completed = metadata.get("completed_steps")
    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise ValueError("checkpoint integrator metadata is missing")
    return PeriodicCheckpointHeader(
        format_version=format_version,
        runtime_identity_sha256=identity,
        completed_steps=completed,
        spectral_refresh_interval=integrator.get("spectral_refresh_interval"),
        integrator_step_count=integrator.get("step_count"),
        integrator_refresh_count=integrator.get("refresh_count"),
        backend_restart=metadata.get("backend_restart"),
        functional_bridge=metadata.get("functional_bridge"),
        compatibility_identity=compatibility_identity,
    )


def _load_tensor(
    directory: Path,
    record: object,
    description: str,
) -> torch.Tensor:
    if not isinstance(record, Mapping):
        raise TypeError(f"{description} tensor record must be a mapping")
    filename = record.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".npy")
    ):
        raise ValueError(f"{description} tensor filename is invalid")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint tensor is missing: {path}")
    expected = _require_sha256(record.get("sha256"), "tensor SHA-256")
    if _sha256(path) != expected:
        raise ValueError(f"checkpoint tensor checksum mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record.get("shape"):
        raise ValueError(f"checkpoint tensor shape metadata differs: {path}")
    if str(values.dtype) != record.get("dtype"):
        raise ValueError(f"checkpoint tensor dtype metadata differs: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"checkpoint tensor contains NaN or Inf: {path}")
    return torch.from_numpy(np.array(values, copy=True))


def load_periodic_checkpoint(
    directory: str | Path,
) -> PeriodicWorkflowCheckpoint:
    """Load and checksum-validate a complete periodic workflow checkpoint."""

    directory = Path(directory).expanduser().resolve()
    metadata = _read_metadata(directory)
    header = read_periodic_checkpoint_header(directory)
    files = metadata.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint tensor-file manifest is missing")
    loaded = {}
    for kind in ("spatial", "spectral"):
        records = files.get(kind)
        if not isinstance(records, Mapping) or tuple(records) != Q_COMPONENTS:
            raise ValueError(f"checkpoint {kind} manifest is incomplete")
        loaded[kind] = {
            name: _load_tensor(directory, records[name], f"{kind}.{name}")
            for name in Q_COMPONENTS
        }
    return PeriodicWorkflowCheckpoint(
        format_version=metadata.get("format_version"),
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=header.spectral_refresh_interval,
        integrator_step_count=header.integrator_step_count,
        integrator_refresh_count=header.integrator_refresh_count,
        evolved_spatial=loaded["spatial"],
        evolved_spectral=loaded["spectral"],
        backend_restart=header.backend_restart,
        functional_bridge=header.functional_bridge,
        compatibility_identity=header.compatibility_identity,
        migration_provenance=metadata.get("migration_provenance"),
    )


def _legacy_periodic_format_for_generation(
    generation: SourceReleaseGeneration,
) -> int:
    if not isinstance(generation, SourceReleaseGeneration):
        raise TypeError("source_release_generation must be a SourceReleaseGeneration")
    return (
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1
        if generation is SourceReleaseGeneration.RC1
        else PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )


def _validate_legacy_periodic_upgrade_contract(
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    source_run_spec: PeriodicBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    if not isinstance(source_run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("source_run_spec must be a PeriodicBerisEdwardsRunSpec")
    expected_format = _legacy_periodic_format_for_generation(
        source_release_generation
    )
    if checkpoint.format_version != expected_format:
        raise ValueError("legacy periodic format does not match source release")
    if checkpoint.functional_bridge is not None:
        raise ValueError(
            "functional periodic checkpoint migration is deferred to RC4.2.4"
        )
    if (
        checkpoint.runtime_identity_sha256
        != source_run_spec.runtime_identity_sha256()
    ):
        raise ValueError(
            "legacy periodic checkpoint opaque identity is not authenticated"
        )
    schema = resolve_legacy_identity_schema(
        LegacyIdentityKey(
            checkpoint_family=CheckpointFamily.PERIODIC_WORKFLOW,
            format_version=expected_format,
            source_release_generation=source_release_generation,
            runtime_path=PERIODIC_RUNTIME_PATH,
            applicability_class="configuration_dependent",
        )
    )
    storage = source_run_spec.simulation.numerics.spectral_storage.value
    if (
        source_release_generation is SourceReleaseGeneration.RC1
        and storage == "full_complex"
    ):
        raise ValueError(
            "rc1 full-complex periodic checkpoint has pre-repair Hermitian dynamics"
        )
    return schema, storage


def upgrade_periodic_checkpoint(
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    source_run_spec: PeriodicBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
) -> PeriodicWorkflowCheckpoint:
    """Create a production v3 copy after explicit legacy adjudication."""

    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be a PeriodicWorkflowCheckpoint")
    if checkpoint.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("current periodic checkpoint does not require upgrade")
    schema, storage = _validate_legacy_periodic_upgrade_contract(
        checkpoint,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    identity = build_periodic_checkpoint_compatibility_identity(
        source_run_spec,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
    )
    return PeriodicWorkflowCheckpoint(
        format_version=PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=checkpoint.runtime_identity_sha256,
        completed_steps=checkpoint.completed_steps,
        spectral_refresh_interval=checkpoint.spectral_refresh_interval,
        integrator_step_count=checkpoint.integrator_step_count,
        integrator_refresh_count=checkpoint.integrator_refresh_count,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
        compatibility_identity=identity,
        migration_provenance={
            "kind": "periodic_checkpoint_identity_upgrade_to_v3",
            "source_release_generation": source_release_generation.value,
            "source_commit": schema.source_commit,
            "source_format_version": checkpoint.format_version,
            "legacy_schema_id": schema.schema_id,
            "legacy_canonicalizer_id": schema.canonicalizer_id,
            "spectral_storage": storage,
            "hermitian_repair_applicability": (
                "inapplicable_to_hermitian_half_storage"
                if source_release_generation is SourceReleaseGeneration.RC1
                else "source_is_post_repair"
            ),
            "source_checkpoint_mutated": False,
        },
    )


def upgrade_periodic_checkpoint_directory(
    source_directory: str | Path,
    target_directory: str | Path,
    *,
    source_run_spec: PeriodicBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
) -> Path:
    """Write an authenticated v3 copy without changing the legacy source."""

    source = Path(source_directory).expanduser().resolve()
    target = Path(target_directory).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("legacy upgrade target must be outside the source checkpoint")
    header = read_periodic_checkpoint_header(source)
    if header.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("current periodic checkpoint does not require upgrade")
    # Authenticate the opaque header and adjudicate dynamics before payload load.
    shell = PeriodicWorkflowCheckpoint(
        format_version=header.format_version,
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=header.spectral_refresh_interval,
        integrator_step_count=header.integrator_step_count,
        integrator_refresh_count=header.integrator_refresh_count,
        evolved_spatial={name: torch.zeros(()) for name in Q_COMPONENTS},
        evolved_spectral={name: torch.zeros(()) for name in Q_COMPONENTS},
        backend_restart=header.backend_restart,
        functional_bridge=header.functional_bridge,
    )
    _validate_legacy_periodic_upgrade_contract(
        shell,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    upgraded = upgrade_periodic_checkpoint(
        load_periodic_checkpoint(source),
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    return write_periodic_checkpoint(target, upgraded)


__all__ = [
    "PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1",
    "PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PeriodicCheckpointHeader",
    "PeriodicWorkflowCheckpoint",
    "build_periodic_checkpoint_compatibility_identity",
    "capture_periodic_checkpoint",
    "load_periodic_checkpoint",
    "periodic_forward_dynamics_identity",
    "read_periodic_checkpoint_header",
    "restore_periodic_checkpoint",
    "upgrade_periodic_checkpoint",
    "upgrade_periodic_checkpoint_directory",
    "write_periodic_checkpoint",
]
