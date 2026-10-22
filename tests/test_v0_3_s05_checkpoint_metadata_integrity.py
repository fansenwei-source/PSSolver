"""S-05 metadata seals, legacy visibility, and non-in-place upgrades."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.channel_active_nematics import (
    CHANNEL_FORWARD_STOKES_NYQUIST_VERSION,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.execution import AlgebraicRuntimeRestartState
from pssolver.experimental import (
    SHADOW_CHECKPOINT_FORMAT_VERSION,
    SHADOW_LEGACY_CHECKPOINT_FORMAT_VERSION,
    ShadowRunCheckpoint,
    load_shadow_checkpoint,
    upgrade_shadow_checkpoint_directory,
    write_shadow_checkpoint,
)
from pssolver.io import (
    CHECKPOINT_INTEGRITY_SEALED_SHA256,
    CHECKPOINT_INTEGRITY_UNSEALED_LEGACY,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    checkpoint_directory_tree_sha256,
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import (
    FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
    FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION,
    PlaneFiniteQAnchoringCheckpoint,
    PlaneRobinScalarCheckpoint,
)
from pssolver.workflows import (
    CHANNEL_UNSEALED_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PLANE_UNSEALED_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    ChannelWorkflowCheckpoint,
    PlaneWorkflowCheckpoint,
    build_plane_checkpoint_compatibility_identity,
    load_channel_checkpoint,
    load_finite_q_anchoring_checkpoint,
    load_plane_checkpoint,
    read_channel_checkpoint_header,
    read_plane_checkpoint_header,
    upgrade_channel_checkpoint_directory,
    upgrade_finite_q_anchoring_checkpoint_directory,
    upgrade_plane_checkpoint_directory_v2,
    write_channel_checkpoint,
    write_finite_q_anchoring_checkpoint,
    write_plane_checkpoint,
)
from pssolver.workflows.channel_checkpoint import CHANNEL_ALGEBRAIC_COMPONENTS


def _tensor(value: float = 0.0) -> torch.Tensor:
    return torch.full((2, 3, 2), value, dtype=torch.float64)


def _plane_checkpoint(tmp_path: Path) -> PlaneWorkflowCheckpoint:
    spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        runtime_path="legacy_production",
        nx=7,
        ny=7,
        nz=6,
        lx=8.0,
        ly=9.0,
        height=20.0,
        dt=0.005,
        steps=3,
        save_start_step=0,
        save_interval=3,
        diagnostic_interval=3,
        spectral_refresh_steps=3,
        seed=24,
        defect_min_separation=2.0,
    )
    spatial = {name: _tensor(index) for index, name in enumerate(Q_COMPONENTS)}
    spectral = {
        name: torch.complex(value, torch.zeros_like(value))
        for name, value in spatial.items()
    }
    backend = {"kind": "s05_test_backend"}
    identity = build_plane_checkpoint_compatibility_identity(
        spec,
        runtime_path=PlaneRuntimePath.LEGACY_PRODUCTION,
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        lifting_restart=None,
    )
    return PlaneWorkflowCheckpoint(
        format_version=PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_path=PlaneRuntimePath.LEGACY_PRODUCTION,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
        completed_steps=2,
        spectral_refresh_interval=3,
        integrator_step_count=2,
        integrator_refresh_count=0,
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        compatibility_identity=identity,
    )


def _channel_checkpoint() -> ChannelWorkflowCheckpoint:
    spatial = {name: _tensor(index) for index, name in enumerate(Q_COMPONENTS)}
    spectral = {
        name: torch.complex(value, torch.zeros_like(value))
        for name, value in spatial.items()
    }
    algebraic = {
        name: _tensor(index + 10)
        for index, name in enumerate(CHANNEL_ALGEBRAIC_COMPONENTS)
    }
    algebraic_spectral = {
        name: torch.complex(value, torch.zeros_like(value))
        for name, value in algebraic.items()
    }
    return ChannelWorkflowCheckpoint(
        format_version=CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_path=ChannelRuntimePath.LEGACY_CHANNEL,
        runtime_identity_sha256="a" * 64,
        completed_steps=2,
        spectral_refresh_interval=3,
        integrator_step_count=2,
        integrator_refresh_count=0,
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        algebraic_spatial=algebraic,
        algebraic_spectral=algebraic_spectral,
        pressure_guess=algebraic_spectral["p"],
        backend_restart={"kind": "s05_test_backend"},
        dynamics_versions={
            "forward_stokes_nyquist": CHANNEL_FORWARD_STOKES_NYQUIST_VERSION
        },
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(torch.tensor(array.shape, dtype=torch.int64).numpy().tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_q_checkpoint() -> PlaneFiniteQAnchoringCheckpoint:
    components = []
    for index, name in enumerate(Q_COMPONENTS):
        remainder = _tensor(index)
        modal = _tensor(index + 1)
        components.append(
            (
                name,
                PlaneRobinScalarCheckpoint(
                    identity_json=json.dumps(
                        {"component": name}, separators=(",", ":"), sort_keys=True
                    ),
                    remainder=remainder,
                    bounded_modal=modal,
                    progress_json='{"completed_steps":2}',
                    representations_json='{"kind":"s05_test"}',
                    remainder_sha256=_tensor_sha256(remainder),
                    bounded_modal_sha256=_tensor_sha256(modal),
                ),
            )
        )
    return PlaneFiniteQAnchoringCheckpoint(
        identity_json='{"kind":"s05_finite_q"}',
        components=tuple(components),
    )


def _shadow_checkpoint() -> ShadowRunCheckpoint:
    spatial = {"Qxx": _tensor(1.0)}
    spectral = {
        "Qxx": torch.complex(spatial["Qxx"], torch.zeros_like(spatial["Qxx"]))
    }
    return ShadowRunCheckpoint(
        format_version=SHADOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256="b" * 64,
        completed_steps=2,
        spectral_refresh_interval=3,
        integrator_step_count=2,
        integrator_refresh_count=0,
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        algebraic_restart=AlgebraicRuntimeRestartState(1, ()),
    )


def _write_current(kind: str, directory: Path, tmp_path: Path) -> Path:
    if kind == "plane":
        return write_plane_checkpoint(directory, _plane_checkpoint(tmp_path))
    if kind == "channel":
        return write_channel_checkpoint(directory, _channel_checkpoint())
    if kind == "finite_q":
        return write_finite_q_anchoring_checkpoint(directory, _finite_q_checkpoint())
    if kind == "shadow":
        return write_shadow_checkpoint(directory, _shadow_checkpoint())
    raise AssertionError(kind)


def _load(kind: str, directory: Path):
    return {
        "plane": load_plane_checkpoint,
        "channel": load_channel_checkpoint,
        "finite_q": load_finite_q_anchoring_checkpoint,
        "shadow": load_shadow_checkpoint,
    }[kind](directory)


def _compatibility_identity(kind: str, checkpoint) -> object:
    if kind == "plane":
        return (
            checkpoint.runtime_identity_sha256,
            checkpoint.compatibility_identity.canonical_sha256(),
        )
    if kind == "channel":
        return checkpoint.runtime_identity_sha256
    if kind == "finite_q":
        return checkpoint.identity_json
    if kind == "shadow":
        return checkpoint.runtime_identity_sha256
    raise AssertionError(kind)


def _downgrade_to_unsealed(kind: str, directory: Path) -> None:
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.pop("metadata_sha256")
    version_key = "schema_version" if kind == "finite_q" else "format_version"
    metadata[version_key] = {
        "plane": PLANE_UNSEALED_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        "channel": CHANNEL_UNSEALED_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        "finite_q": FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION,
        "shadow": SHADOW_LEGACY_CHECKPOINT_FORMAT_VERSION,
    }[kind]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coordinated_progress_tamper(kind: str, metadata: dict[str, object]) -> None:
    metadata["completed_steps"] = 3
    if kind == "finite_q":
        for component in metadata["components"].values():
            component["progress"]["completed_steps"] = 3
        return
    metadata["integrator"] = {
        "spectral_refresh_interval": 3,
        "step_count": 0,
        "refresh_count": 1,
    }


def _metadata_tamper(
    kind: str,
    mutation: str,
    metadata: dict[str, object],
) -> None:
    if mutation == "tensor_record":
        if kind == "finite_q":
            metadata["components"]["Qxx"]["tensors"]["remainder"]["file"][
                "sha256"
            ] = "0" * 64
        else:
            metadata["tensor_files"]["evolved_spatial"]["Qxx"]["sha256"] = (
                "0" * 64
            )
        return
    if mutation == "restart_or_persistent_state":
        if kind in {"plane", "channel"}:
            metadata["backend_restart"] = {"kind": "tampered"}
        elif kind == "finite_q":
            metadata["components"]["Qxx"]["representations"] = {
                "kind": "tampered"
            }
        else:
            metadata["algebraic_restart"]["format_version"] = 999
        return
    if mutation == "identity":
        key = {
            "plane": "compatibility_identity_sha256",
            "channel": "runtime_identity_sha256",
            "finite_q": "identity_sha256",
            "shadow": "runtime_identity_sha256",
        }[kind]
        metadata[key] = "0" * 64
        return
    if mutation == "unknown_key":
        metadata["unregistered_field"] = True
        return
    if mutation == "missing_key":
        metadata.pop("completed_steps")
        return
    raise AssertionError(mutation)


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
def test_current_formats_are_sealed_and_tamper_before_payload_load(
    tmp_path, monkeypatch, kind
):
    directory = _write_current(kind, tmp_path / kind, tmp_path)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert verify_checkpoint_metadata(metadata, required=True)
    metadata["completed_steps"] += 1
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if kind == "plane":
        monkeypatch.setattr(
            "pssolver.workflows.plane_checkpoint._load_tensor",
            lambda *_args, **_kwargs: pytest.fail("payload load started"),
        )
    elif kind == "channel":
        monkeypatch.setattr(
            "pssolver.workflows.channel_checkpoint._load_tensor",
            lambda *_args, **_kwargs: pytest.fail("payload load started"),
        )
    elif kind == "finite_q":
        monkeypatch.setattr(
            "pssolver.workflows.finite_q_anchoring._load_tensor",
            lambda *_args, **_kwargs: pytest.fail("payload load started"),
        )
    else:
        monkeypatch.setattr(
            "pssolver.experimental.checkpointing._load_tensor_record",
            lambda *_args, **_kwargs: pytest.fail("payload load started"),
        )
    with pytest.raises(CheckpointIntegrityError, match="metadata checksum mismatch"):
        _load(kind, directory)


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
def test_coordinated_progress_tamper_without_recomputed_digest_is_rejected(
    tmp_path, kind
):
    directory = _write_current(kind, tmp_path / kind, tmp_path)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _coordinated_progress_tamper(kind, metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointIntegrityError, match="metadata checksum mismatch"):
        _load(kind, directory)


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
@pytest.mark.parametrize(
    ("replacement", "message"),
    ((None, "SHA-256 is missing"), ("invalid", "SHA-256 is invalid")),
)
def test_current_formats_require_a_well_formed_metadata_digest(
    tmp_path, kind, replacement, message
):
    directory = _write_current(kind, tmp_path / kind, tmp_path)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if replacement is None:
        metadata.pop("metadata_sha256")
    else:
        metadata["metadata_sha256"] = replacement
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointIntegrityError, match=message):
        _load(kind, directory)


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
@pytest.mark.parametrize(
    "mutation",
    (
        "tensor_record",
        "restart_or_persistent_state",
        "identity",
        "unknown_key",
        "missing_key",
    ),
)
def test_every_registered_metadata_class_is_covered_by_the_seal(
    tmp_path, kind, mutation
):
    directory = _write_current(kind, tmp_path / kind, tmp_path)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _metadata_tamper(kind, mutation, metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointIntegrityError, match="metadata checksum mismatch"):
        _load(kind, directory)


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
@pytest.mark.parametrize("mutation", ("unknown_key", "missing_key"))
def test_exact_current_schema_still_rejects_after_digest_recomputation(
    tmp_path, kind, mutation
):
    directory = _write_current(kind, tmp_path / kind, tmp_path)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("metadata_sha256")
    _metadata_tamper(kind, mutation, metadata)
    metadata_path.write_text(
        json.dumps(seal_checkpoint_metadata(metadata), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CheckpointIntegrityError, match="schema differs"):
        _load(kind, directory)


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
def test_unsealed_formats_are_read_only_visible_and_upgrade_with_honest_source(
    tmp_path, kind
):
    source = _write_current(kind, tmp_path / f"{kind}_source", tmp_path)
    _downgrade_to_unsealed(kind, source)
    source_tree_sha256 = checkpoint_directory_tree_sha256(source)
    loaded = _load(kind, source)
    assert loaded.integrity_status == CHECKPOINT_INTEGRITY_UNSEALED_LEGACY

    with pytest.raises(CheckpointCompatibilityError, match="read-only"):
        {
            "plane": write_plane_checkpoint,
            "channel": write_channel_checkpoint,
            "finite_q": write_finite_q_anchoring_checkpoint,
            "shadow": write_shadow_checkpoint,
        }[kind](tmp_path / f"{kind}_forbidden", loaded)

    target = tmp_path / f"{kind}_upgraded"
    {
        "plane": upgrade_plane_checkpoint_directory_v2,
        "channel": upgrade_channel_checkpoint_directory,
        "finite_q": upgrade_finite_q_anchoring_checkpoint_directory,
        "shadow": upgrade_shadow_checkpoint_directory,
    }[kind](source, target)
    assert checkpoint_directory_tree_sha256(source) == source_tree_sha256
    upgraded = _load(kind, target)
    assert upgraded.integrity_status == CHECKPOINT_INTEGRITY_SEALED_SHA256
    assert _compatibility_identity(kind, upgraded) == _compatibility_identity(
        kind, loaded
    )
    provenance = upgraded.migration_provenance
    assert provenance["source_integrity_status"] == "unsealed_legacy"
    assert provenance["source_tree_sha256"] == source_tree_sha256
    assert provenance["source_historical_integrity_authenticated"] is False
    assert provenance["source_checkpoint_mutated"] is False


def test_header_integrity_status_is_outside_compatibility_identity(tmp_path):
    plane = write_plane_checkpoint(
        tmp_path / "plane",
        _plane_checkpoint(tmp_path),
    )
    channel = write_channel_checkpoint(
        tmp_path / "channel",
        _channel_checkpoint(),
    )
    plane_header = read_plane_checkpoint_header(plane)
    channel_header = read_channel_checkpoint_header(channel)
    assert plane_header.integrity_status == CHECKPOINT_INTEGRITY_SEALED_SHA256
    assert channel_header.integrity_status == CHECKPOINT_INTEGRITY_SEALED_SHA256
    assert "integrity_status" not in json.loads(
        (plane / "checkpoint.json").read_text(encoding="utf-8")
    )["compatibility_identity"]
    assert "integrity_status" not in json.loads(
        (channel / "checkpoint.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("kind", ("plane", "channel", "finite_q", "shadow"))
def test_sealed_upgrade_requires_honest_complete_source_provenance(
    tmp_path, kind
):
    checkpoint = {
        "plane": lambda: _plane_checkpoint(tmp_path),
        "channel": _channel_checkpoint,
        "finite_q": _finite_q_checkpoint,
        "shadow": _shadow_checkpoint,
    }[kind]()
    incomplete = replace(
        checkpoint,
        migration_provenance={
            "kind": f"{kind}_legacy_upgrade",
            "source_integrity_status": "unsealed_legacy",
            "source_historical_integrity_authenticated": False,
            "source_checkpoint_mutated": False,
        },
    )
    writer = {
        "plane": write_plane_checkpoint,
        "channel": write_channel_checkpoint,
        "finite_q": write_finite_q_anchoring_checkpoint,
        "shadow": write_shadow_checkpoint,
    }[kind]
    with pytest.raises(
        CheckpointCompatibilityError,
        match="migration provenance is incomplete",
    ):
        writer(tmp_path / f"{kind}_incomplete_upgrade", incomplete)
