"""Versioned exact-restart checkpoints for the dual-path Plane workflow."""

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

from pssolver.configuration import PlaneBerisEdwardsRunSpec, PlaneRuntimePath
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
from pssolver.runtime import (
    PlaneRuntimeAdapterProtocol,
    plane_lifting_restart_metadata,
    verify_plane_lifting_identity,
)


PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 1
PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 2


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


@dataclass(frozen=True, slots=True)
class PlaneCheckpointHeader:
    """Small header that can reject cross-backend restart before arrays load."""

    format_version: int
    runtime_path: PlaneRuntimePath
    completed_steps: int
    runtime_identity_sha256: str | None = None
    compatibility_identity: CheckpointCompatibilityIdentity | None = None

    @property
    def compatibility_identity_sha256(self) -> str | None:
        identity = self.compatibility_identity
        return None if identity is None else identity.canonical_sha256()


@dataclass(frozen=True, slots=True)
class PlaneWorkflowCheckpoint:
    """Complete evolved state and refresh phase for one Plane backend."""

    format_version: int
    runtime_path: PlaneRuntimePath
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    evolved_spatial: Mapping[str, torch.Tensor]
    evolved_spectral: Mapping[str, torch.Tensor]
    backend_restart: Mapping[str, object]
    lifting_restart: Mapping[str, object] | None = None
    compatibility_identity: CheckpointCompatibilityIdentity | None = None
    migration_provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.format_version not in {
            PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        }:
            raise ValueError("unsupported Plane workflow checkpoint version")
        if not isinstance(self.runtime_path, PlaneRuntimePath):
            raise TypeError("checkpoint runtime_path must be PlaneRuntimePath")
        _require_sha256(
            self.runtime_identity_sha256,
            "runtime_identity_sha256",
        )
        for value, description in (
            (self.completed_steps, "completed_steps"),
            (self.integrator_step_count, "integrator_step_count"),
            (self.integrator_refresh_count, "integrator_refresh_count"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{description} must be non-negative")
        interval = self.spectral_refresh_interval
        if interval is not None and (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ValueError("spectral refresh interval must be positive or None")
        if interval is None:
            counters_match = (
                self.integrator_refresh_count == 0
                and self.integrator_step_count == self.completed_steps
            )
        else:
            counters_match = (
                self.integrator_step_count < interval
                and self.integrator_refresh_count * interval
                + self.integrator_step_count
                == self.completed_steps
            )
        if not counters_match:
            raise ValueError("checkpoint refresh counters are inconsistent")
        spatial = _clone_tensors(self.evolved_spatial, "evolved_spatial")
        spectral = _clone_tensors(self.evolved_spectral, "evolved_spectral")
        if not isinstance(self.backend_restart, Mapping):
            raise TypeError("backend_restart must be a mapping")
        backend = dict(self.backend_restart)
        try:
            json.dumps(backend, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("backend restart metadata is not JSON-compatible") from exc
        object.__setattr__(self, "evolved_spatial", spatial)
        object.__setattr__(self, "evolved_spectral", spectral)
        object.__setattr__(self, "backend_restart", MappingProxyType(backend))
        lifting = self.lifting_restart
        if lifting is not None:
            if not isinstance(lifting, Mapping):
                raise TypeError("lifting_restart must be a mapping or None")
            try:
                encoded = json.dumps(
                    dict(lifting),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "lifting restart metadata is not JSON-compatible"
                ) from exc
            object.__setattr__(
                self,
                "lifting_restart",
                MappingProxyType(json.loads(encoded)),
            )
        identity = self.compatibility_identity
        if self.format_version == PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            if identity is not None:
                raise ValueError("legacy Plane checkpoints cannot carry v2 identity")
        else:
            if not isinstance(identity, CheckpointCompatibilityIdentity):
                raise TypeError("format-v2 Plane checkpoint identity is missing")
            if identity.family is not CheckpointFamily.PLANE_WORKFLOW:
                raise ValueError("Plane checkpoint identity has the wrong family")
            if identity.runtime_path != self.runtime_path.value:
                raise ValueError("Plane checkpoint identity runtime path differs")
        migration = self.migration_provenance
        if migration is not None:
            if not isinstance(migration, Mapping):
                raise TypeError("migration_provenance must be a mapping or None")
            try:
                encoded = json.dumps(
                    dict(migration),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "migration provenance is not JSON-compatible"
                ) from exc
            object.__setattr__(
                self,
                "migration_provenance",
                MappingProxyType(json.loads(encoded)),
            )

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format_version": self.format_version,
            "runtime_path": self.runtime_path.value,
            "completed_steps": self.completed_steps,
            "integrator": {
                "spectral_refresh_interval": self.spectral_refresh_interval,
                "step_count": self.integrator_step_count,
                "refresh_count": self.integrator_refresh_count,
            },
            "evolved_components": list(Q_COMPONENTS),
            "backend_restart": dict(self.backend_restart),
        }
        if self.format_version == PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            metadata["runtime_identity_sha256"] = self.runtime_identity_sha256
        else:
            identity = self.compatibility_identity
            if identity is None:  # pragma: no cover - guarded by __post_init__
                raise RuntimeError("format-v2 identity unexpectedly missing")
            metadata["compatibility_identity"] = identity.to_metadata()
            metadata["compatibility_identity_sha256"] = (
                identity.canonical_sha256()
            )
            metadata["run_provenance"] = {
                "legacy_runtime_identity_sha256": (
                    self.runtime_identity_sha256
                )
            }
        if self.lifting_restart is not None:
            metadata["lifting_restart"] = dict(self.lifting_restart)
        if self.migration_provenance is not None:
            metadata["migration_provenance"] = dict(
                self.migration_provenance
            )
        return metadata


def _plane_run_spec_dynamics(
    run_spec: PlaneBerisEdwardsRunSpec,
) -> dict[str, object]:
    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    metadata = json.loads(
        json.dumps(
            run_spec.runtime_identity_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    metadata.pop("runtime_path")
    controls = metadata["runtime_controls"]
    controls.pop("device")
    return metadata


def plane_forward_dynamics_identity(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    lifting_restart: Mapping[str, object] | None,
) -> CheckpointIdentityLayer:
    """Build the Plane next-step identity without run provenance."""

    lifting = _lifting_compatibility_identity(lifting_restart)
    run_spec_dynamics = plane_run_spec_dynamics_identity(run_spec)
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.FORWARD_DYNAMICS,
        version="plane.forward.v2",
        payload={
            "run_spec_dynamics": run_spec_dynamics.to_metadata(),
            "lifting_dynamics": lifting,
            "lifting_equation_version": (
                "none" if lifting is None else "static_affine_dirichlet.v1"
            ),
        },
    )


def plane_run_spec_dynamics_identity(
    run_spec: PlaneBerisEdwardsRunSpec,
) -> CheckpointIdentityLayer:
    """Return the allocation-free part of Plane forward dynamics."""

    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.FORWARD_DYNAMICS,
        version="plane.run_spec_dynamics.v2",
        payload={
            "run_spec": _plane_run_spec_dynamics(run_spec),
            "dynamics_versions": {
                "q_evolution": "complete_stress_beris_edwards.v1",
                "stokes_nyquist": "plane_free_slip.periodic_nyquist_zero.v2",
                "spectral_refresh": "semi_implicit_euler_refresh.v1",
            },
        },
    )


def _plane_state_layout_identity(
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
        version="plane.state.v2",
        payload={
            "component_order": list(Q_COMPONENTS),
            "spatial": layout(evolved_spatial),
            "spectral": layout(evolved_spectral),
        },
    )


def _plane_backend_restart_identity(
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
    for provenance_key in (
        "runtime_identity_sha256",
        "format_v1_runtime_path_carrier",
        "compiled_runtime_identity",
    ):
        normalized.pop(provenance_key, None)
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.BACKEND_RESTART,
        version="plane.backend.v1",
        payload=normalized,
    )


def build_plane_checkpoint_compatibility_identity(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    runtime_path: PlaneRuntimePath,
    evolved_spatial: Mapping[str, torch.Tensor],
    evolved_spectral: Mapping[str, torch.Tensor],
    backend_restart: Mapping[str, object],
    lifting_restart: Mapping[str, object] | None,
) -> CheckpointCompatibilityIdentity:
    if not isinstance(runtime_path, PlaneRuntimePath):
        raise TypeError("runtime_path must be a PlaneRuntimePath")
    if run_spec.runtime_path is not runtime_path:
        raise ValueError("run specification and checkpoint runtime path differ")
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.PLANE_WORKFLOW,
        runtime_path=runtime_path.value,
        forward_dynamics=plane_forward_dynamics_identity(
            run_spec,
            lifting_restart=lifting_restart,
        ),
        state_layout=_plane_state_layout_identity(
            evolved_spatial,
            evolved_spectral,
        ),
        backend_restart=_plane_backend_restart_identity(backend_restart),
    )


def _validate_plane_checkpoint_embedded_identity(
    checkpoint: PlaneWorkflowCheckpoint,
) -> None:
    """Bind persisted layout/backend/lifting metadata to the signed layers."""

    identity = checkpoint.compatibility_identity
    if identity is None:
        raise ValueError("format-v2 Plane checkpoint identity is missing")
    expected_layout = _plane_state_layout_identity(
        checkpoint.evolved_spatial,
        checkpoint.evolved_spectral,
    )
    if identity.state_layout != expected_layout:
        raise ValueError("checkpoint state layout identity is inconsistent")
    expected_backend = _plane_backend_restart_identity(
        checkpoint.backend_restart
    )
    if identity.backend_restart != expected_backend:
        raise ValueError("checkpoint backend restart identity is inconsistent")
    forward_payload = identity.forward_dynamics.to_metadata()["payload"]
    if forward_payload.get("lifting_dynamics") != (
        _lifting_compatibility_identity(checkpoint.lifting_restart)
    ):
        raise ValueError("checkpoint lifting identity is inconsistent")


def capture_plane_checkpoint(
    adapter: PlaneRuntimeAdapterProtocol,
    *,
    run_spec: PlaneBerisEdwardsRunSpec,
) -> PlaneWorkflowCheckpoint:
    """Synchronize algebraic fields and capture layered format-v2 state."""

    if not isinstance(adapter, PlaneRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")
    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    runtime_identity_sha256 = run_spec.runtime_identity_sha256()
    adapter.synchronize_for_observation()
    fields = adapter.fields
    integrator = adapter.solver.integrator
    spatial = {name: fields[name] for name in Q_COMPONENTS}
    spectral = {name: fields[f"{name}.hat"] for name in Q_COMPONENTS}
    backend = adapter.backend_restart_metadata()
    lifting = plane_lifting_restart_metadata(adapter)
    compatibility_identity = build_plane_checkpoint_compatibility_identity(
        run_spec,
        runtime_path=adapter.runtime_path,
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        lifting_restart=lifting,
    )
    return PlaneWorkflowCheckpoint(
        format_version=PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_path=adapter.runtime_path,
        runtime_identity_sha256=runtime_identity_sha256,
        completed_steps=adapter.completed_steps,
        spectral_refresh_interval=integrator.spectral_refresh_interval,
        integrator_step_count=int(integrator.step_count),
        integrator_refresh_count=int(integrator.refresh_count),
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        lifting_restart=lifting,
        compatibility_identity=compatibility_identity,
    )


def restore_plane_checkpoint(
    adapter: PlaneRuntimeAdapterProtocol,
    checkpoint: PlaneWorkflowCheckpoint,
    *,
    run_spec: PlaneBerisEdwardsRunSpec,
) -> int:
    """Restore a checksum-validated checkpoint on its exact runtime path."""

    if not isinstance(adapter, PlaneRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")
    if not isinstance(checkpoint, PlaneWorkflowCheckpoint):
        raise TypeError("checkpoint must be PlaneWorkflowCheckpoint")
    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    if checkpoint.format_version == PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "legacy Plane checkpoint requires explicit identity upgrade"
        )
    if checkpoint.runtime_path is not adapter.runtime_path:
        raise ValueError("cross-runtime Plane checkpoint restart is unsupported")
    _validate_plane_checkpoint_embedded_identity(checkpoint)
    expected_lifting = plane_lifting_restart_metadata(adapter)
    observed_lifting = (
        None
        if checkpoint.lifting_restart is None
        else dict(checkpoint.lifting_restart)
    )
    if _lifting_compatibility_identity(observed_lifting) != (
        _lifting_compatibility_identity(expected_lifting)
    ):
        raise ValueError("checkpoint static lifting identity does not match target")

    fields = adapter.fields
    target_spatial = {name: fields[name] for name in Q_COMPONENTS}
    target_spectral = {
        name: fields[f"{name}.hat"] for name in Q_COMPONENTS
    }
    target_identity = build_plane_checkpoint_compatibility_identity(
        run_spec,
        runtime_path=adapter.runtime_path,
        evolved_spatial=target_spatial,
        evolved_spectral=target_spectral,
        backend_restart=adapter.backend_restart_metadata(),
        lifting_restart=expected_lifting,
    )
    if checkpoint.compatibility_identity != target_identity:
        raise ValueError(
            "checkpoint compatibility identity does not match target"
        )
    verify_plane_lifting_identity(adapter)

    targets: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint shape for {name}{suffix} differs")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
            targets.append((target, source, f"{name}{suffix}"))

    # All identities, shapes, and dtypes are validated before the first target
    # mutation.  Device transfer happens only after that fail-closed boundary.
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


def _lifting_compatibility_identity(
    metadata: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Remove execution-location provenance from lifting compatibility.

    Older checkpoints stored ``device`` inside the lifting and linear-
    correction records.  Those locations do not alter the numerical lifting
    plan or checkpoint tensors, so accept them across CPU/CUDA materialization
    while retaining every scientific identity field.
    """

    if metadata is None:
        return None
    normalized = json.loads(
        json.dumps(
            dict(metadata),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    normalized.pop("materialization_provenance", None)
    lifting = normalized.get("lifting")
    if isinstance(lifting, dict):
        lifting.pop("device", None)
        lifting.pop("fresh_initial_remainder_conditioning", None)
    corrections = normalized.get("linear_corrections", ())
    if isinstance(corrections, list):
        for correction in corrections:
            if isinstance(correction, dict):
                correction.pop("device", None)
    return normalized


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


def write_plane_checkpoint(
    directory: str | Path,
    checkpoint: PlaneWorkflowCheckpoint,
) -> Path:
    """Atomically write a new checkpoint directory without pickle."""

    if not isinstance(checkpoint, PlaneWorkflowCheckpoint):
        raise TypeError("checkpoint must be PlaneWorkflowCheckpoint")
    target = Path(directory).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        files = {"evolved_spatial": {}, "evolved_spectral": {}}
        for kind, tensors in (
            ("evolved_spatial", checkpoint.evolved_spatial),
            ("evolved_spectral", checkpoint.evolved_spectral),
        ):
            for name, value in tensors.items():
                files[kind][name] = _write_tensor(
                    staging / f"{kind}__{name}.npy",
                    value,
                )
        metadata = checkpoint.to_metadata()
        metadata["tensor_files"] = files
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
    return metadata


def read_plane_checkpoint_header(directory: str | Path) -> PlaneCheckpointHeader:
    """Read only the versioned identity fields needed for an early gate."""

    metadata = _read_metadata(Path(directory).expanduser().resolve())
    format_version = metadata.get("format_version")
    if format_version not in {
        PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    }:
        raise ValueError("unsupported Plane workflow checkpoint version")
    try:
        runtime_path = PlaneRuntimePath(metadata.get("runtime_path"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint runtime path is invalid") from exc
    completed = metadata.get("completed_steps")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise ValueError("checkpoint completed_steps is invalid")
    if format_version == PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        runtime_identity = _require_sha256(
            metadata.get("runtime_identity_sha256"),
            "runtime_identity_sha256",
        )
        compatibility_identity = None
    else:
        run_provenance = metadata.get("run_provenance")
        if not isinstance(run_provenance, Mapping):
            raise ValueError("checkpoint run provenance is missing")
        runtime_identity = _require_sha256(
            run_provenance.get("legacy_runtime_identity_sha256"),
            "legacy_runtime_identity_sha256",
        )
        compatibility_identity = CheckpointCompatibilityIdentity.from_metadata(
            metadata.get("compatibility_identity")
        )
        expected = _require_sha256(
            metadata.get("compatibility_identity_sha256"),
            "compatibility_identity_sha256",
        )
        if compatibility_identity.canonical_sha256() != expected:
            raise ValueError("checkpoint compatibility identity checksum mismatch")
        if compatibility_identity.family is not CheckpointFamily.PLANE_WORKFLOW:
            raise ValueError("checkpoint compatibility identity family differs")
        if compatibility_identity.runtime_path != runtime_path.value:
            raise ValueError("checkpoint compatibility runtime path differs")
    return PlaneCheckpointHeader(
        format_version=format_version,
        runtime_path=runtime_path,
        completed_steps=completed,
        runtime_identity_sha256=runtime_identity,
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


def load_plane_checkpoint(directory: str | Path) -> PlaneWorkflowCheckpoint:
    """Load and checksum-validate a complete Plane workflow checkpoint."""

    directory = Path(directory).expanduser().resolve()
    metadata = _read_metadata(directory)
    header = read_plane_checkpoint_header(directory)
    files = metadata.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint tensor-file manifest is missing")
    loaded = {}
    for kind in ("evolved_spatial", "evolved_spectral"):
        records = files.get(kind)
        if not isinstance(records, Mapping) or tuple(records) != Q_COMPONENTS:
            raise ValueError(f"checkpoint {kind} manifest is incomplete")
        loaded[kind] = {
            name: _load_tensor(directory, records[name], f"{kind}.{name}")
            for name in Q_COMPONENTS
        }
    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise ValueError("checkpoint integrator metadata is missing")
    backend = metadata.get("backend_restart")
    if not isinstance(backend, Mapping):
        raise ValueError("checkpoint backend restart metadata is missing")
    return PlaneWorkflowCheckpoint(
        format_version=metadata.get("format_version"),
        runtime_path=header.runtime_path,
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=integrator.get("spectral_refresh_interval"),
        integrator_step_count=integrator.get("step_count"),
        integrator_refresh_count=integrator.get("refresh_count"),
        evolved_spatial=loaded["evolved_spatial"],
        evolved_spectral=loaded["evolved_spectral"],
        backend_restart=backend,
        lifting_restart=metadata.get("lifting_restart"),
        compatibility_identity=header.compatibility_identity,
        migration_provenance=metadata.get("migration_provenance"),
    )


def _legacy_plane_registry_key(
    *,
    generation: SourceReleaseGeneration,
    runtime_path: PlaneRuntimePath,
) -> LegacyIdentityKey:
    if not isinstance(generation, SourceReleaseGeneration):
        raise TypeError("generation must be a SourceReleaseGeneration")
    return LegacyIdentityKey(
        checkpoint_family=CheckpointFamily.PLANE_WORKFLOW,
        format_version=PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        source_release_generation=generation,
        runtime_path=runtime_path.value,
        applicability_class="configuration_dependent",
    )


def _validate_legacy_plane_upgrade_contract(
    *,
    runtime_path: PlaneRuntimePath,
    runtime_identity_sha256: str,
    lifting_restart: Mapping[str, object] | None,
    source_run_spec: PlaneBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    if not isinstance(source_run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("source_run_spec must be a PlaneBerisEdwardsRunSpec")
    schema = resolve_legacy_identity_schema(
        _legacy_plane_registry_key(
            generation=source_release_generation,
            runtime_path=runtime_path,
        )
    )
    if source_run_spec.runtime_path is not runtime_path:
        raise ValueError("legacy source run specification path differs")
    if source_run_spec.runtime_identity_sha256() != runtime_identity_sha256:
        raise ValueError("legacy checkpoint opaque identity is not authenticated")
    if (
        source_release_generation is SourceReleaseGeneration.RC1
        and lifting_restart is not None
    ):
        raise ValueError(
            "rc1 lifted Plane checkpoint has incompatible lifting dynamics"
        )
    rc4_1_nyquist_applicable = (
        source_run_spec.nx % 2 == 0 or source_run_spec.ny % 2 == 0
    )
    if rc4_1_nyquist_applicable:
        raise ValueError(
            "legacy Plane checkpoint has pre-RC4.1 Nyquist dynamics"
        )
    return schema


def upgrade_plane_checkpoint_v1(
    checkpoint: PlaneWorkflowCheckpoint,
    *,
    source_run_spec: PlaneBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
) -> PlaneWorkflowCheckpoint:
    """Create a v2 checkpoint after explicit legacy dynamics adjudication."""

    if not isinstance(checkpoint, PlaneWorkflowCheckpoint):
        raise TypeError("checkpoint must be a PlaneWorkflowCheckpoint")
    if checkpoint.format_version != PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("only Plane checkpoint format v1 can be upgraded")
    schema = _validate_legacy_plane_upgrade_contract(
        runtime_path=checkpoint.runtime_path,
        runtime_identity_sha256=checkpoint.runtime_identity_sha256,
        lifting_restart=checkpoint.lifting_restart,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    identity = build_plane_checkpoint_compatibility_identity(
        source_run_spec,
        runtime_path=checkpoint.runtime_path,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
        lifting_restart=checkpoint.lifting_restart,
    )
    return PlaneWorkflowCheckpoint(
        format_version=PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_path=checkpoint.runtime_path,
        runtime_identity_sha256=checkpoint.runtime_identity_sha256,
        completed_steps=checkpoint.completed_steps,
        spectral_refresh_interval=checkpoint.spectral_refresh_interval,
        integrator_step_count=checkpoint.integrator_step_count,
        integrator_refresh_count=checkpoint.integrator_refresh_count,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
        lifting_restart=checkpoint.lifting_restart,
        compatibility_identity=identity,
        migration_provenance={
            "kind": "plane_checkpoint_identity_upgrade_v1_to_v2",
            "source_release_generation": source_release_generation.value,
            "source_commit": schema.source_commit,
            "legacy_schema_id": schema.schema_id,
            "legacy_canonicalizer_id": schema.canonicalizer_id,
            "legacy_runtime_identity_sha256": (
                checkpoint.runtime_identity_sha256
            ),
            "rc4_1_nyquist_applicable": False,
            "rc4_1_nyquist_applicability_basis": (
                "both_periodic_axis_lengths_are_odd"
            ),
            "source_checkpoint_mutated": False,
        },
    )


def upgrade_plane_checkpoint_directory_v1(
    source_directory: str | Path,
    target_directory: str | Path,
    *,
    source_run_spec: PlaneBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
) -> Path:
    """Write an authenticated v2 copy without modifying the v1 source."""

    source = Path(source_directory).expanduser().resolve()
    target_directory = Path(target_directory).expanduser().resolve()
    if target_directory == source or source in target_directory.parents:
        raise ValueError(
            "legacy upgrade target must be outside the source checkpoint"
        )
    header = read_plane_checkpoint_header(source)
    if header.format_version != PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("only Plane checkpoint format v1 can be upgraded")
    metadata = _read_metadata(source)
    lifting = metadata.get("lifting_restart")
    if lifting is not None and not isinstance(lifting, Mapping):
        raise ValueError("legacy lifting restart metadata is invalid")
    _validate_legacy_plane_upgrade_contract(
        runtime_path=header.runtime_path,
        runtime_identity_sha256=header.runtime_identity_sha256,
        lifting_restart=lifting,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    source_hashes = {
        path.relative_to(source).as_posix(): _sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    checkpoint = load_plane_checkpoint(source)
    upgraded = upgrade_plane_checkpoint_v1(
        checkpoint,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    target = write_plane_checkpoint(target_directory, upgraded)
    final_hashes = {
        path.relative_to(source).as_posix(): _sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    if source_hashes != final_hashes:
        raise RuntimeError("legacy source checkpoint changed during upgrade")
    return target


__all__ = [
    "PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PlaneCheckpointHeader",
    "PlaneWorkflowCheckpoint",
    "build_plane_checkpoint_compatibility_identity",
    "capture_plane_checkpoint",
    "load_plane_checkpoint",
    "plane_forward_dynamics_identity",
    "plane_run_spec_dynamics_identity",
    "read_plane_checkpoint_header",
    "restore_plane_checkpoint",
    "upgrade_plane_checkpoint_directory_v1",
    "upgrade_plane_checkpoint_v1",
    "write_plane_checkpoint",
]
