"""RC4.2.2 Plane layered checkpoint identity and legacy upgrade gates."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.io.checkpoint_identity import SourceReleaseGeneration
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows import plane_checkpoint as checkpoint_module
from pssolver.workflows.plane_checkpoint import (
    PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    build_plane_checkpoint_compatibility_identity,
    capture_plane_checkpoint,
    load_plane_checkpoint,
    read_plane_checkpoint_header,
    restore_plane_checkpoint,
    upgrade_plane_checkpoint_directory_v1,
    upgrade_plane_checkpoint_v1,
    write_plane_checkpoint,
)


def _spec(
    tmp_path: Path,
    name: str,
    *,
    nx: int = 8,
    ny: int = 8,
    device: str = "cpu",
    seed: int = 24,
    dt: float = 0.005,
):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / name,
        device=device,
        dtype="float64",
        pointwise_execution="eager",
        runtime_path="legacy_production",
        nx=nx,
        ny=ny,
        nz=6,
        lx=8.0,
        ly=9.0,
        height=20.0,
        dt=dt,
        steps=8,
        save_start_step=0,
        save_interval=8,
        diagnostic_interval=8,
        spectral_refresh_steps=3,
        seed=seed,
        defect_min_separation=2.0,
    )


def _initial_values(nx: int, ny: int):
    coordinate = torch.arange(
        nx * ny * 6,
        dtype=torch.float64,
    ).reshape(nx, ny, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _runtime(spec):
    solver, projector = build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(spec.nx, spec.ny),
    )
    return LegacyPlaneRuntimeAdapter(solver, projector)


def _state(runtime):
    return {
        f"{name}{suffix}": runtime.fields[f"{name}{suffix}"].clone()
        for name in Q_COMPONENTS
        for suffix in ("", ".hat")
    }


def _assert_state_equal(left, right) -> None:
    left_state = _state(left)
    right_state = _state(right)
    assert left_state.keys() == right_state.keys()
    assert all(
        torch.equal(value, right_state[name])
        for name, value in left_state.items()
    )


def _tree_sha256(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _synthetic_lifting(plan_sha256: str, conditioning: str):
    return {
        "schema_version": 1,
        "representation": "homogeneous_remainder",
        "physical_reconstruction": "homogeneous_remainder_plus_lift",
        "lifting": {
            "plan_sha256": plan_sha256,
            "component_order": list(Q_COMPONENTS),
            "shape": [8, 8, 6],
            "dtype": "torch.float64",
            "materialized_lift_sha256": "c" * 64,
            "affine_laplacian_explicit": True,
            "allocation_lifetime": "construction_owned_static",
            "fresh_initial_remainder_conditioning": {
                "kind": conditioning,
            },
        },
        "linear_corrections": [],
        "convention": {"kind": "test_static_lift"},
        "materialization_provenance": {"lifting_device": "cpu"},
    }


def _checkpoint_with_lifting(checkpoint, spec, lifting):
    identity = build_plane_checkpoint_compatibility_identity(
        spec,
        runtime_path=checkpoint.runtime_path,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
        lifting_restart=lifting,
    )
    return replace(
        checkpoint,
        lifting_restart=lifting,
        compatibility_identity=identity,
    )


def _legacy_checkpoint(runtime, spec):
    current = capture_plane_checkpoint(runtime, run_spec=spec)
    return replace(
        current,
        format_version=PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        compatibility_identity=None,
        migration_provenance=None,
    )


def test_m01_device_token_is_provenance_not_restart_compatibility(tmp_path):
    source_spec = _spec(tmp_path, "source", device="cpu")
    target_spec = _spec(tmp_path, "target", device="auto")
    source = _runtime(source_spec)
    target = _runtime(target_spec)
    source.advance(2)

    checkpoint = capture_plane_checkpoint(source, run_spec=source_spec)

    assert source_spec.runtime_identity_sha256() != (
        target_spec.runtime_identity_sha256()
    )
    assert restore_plane_checkpoint(
        target,
        checkpoint,
        run_spec=target_spec,
    ) == 2
    _assert_state_equal(source, target)


def test_m02_fresh_initializer_and_conditioning_are_not_restart_identity(
    tmp_path,
    monkeypatch,
):
    source_spec = _spec(tmp_path, "source", seed=24)
    target_spec = _spec(tmp_path, "target", seed=91)
    source = _runtime(source_spec)
    target = _runtime(target_spec)
    source.advance(1)
    checkpoint = capture_plane_checkpoint(source, run_spec=source_spec)
    source_lifting = _synthetic_lifting("a" * 64, "source_initializer")
    target_lifting = _synthetic_lifting("a" * 64, "target_initializer")
    checkpoint = _checkpoint_with_lifting(
        checkpoint,
        source_spec,
        source_lifting,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "plane_lifting_restart_metadata",
        lambda _adapter: target_lifting,
    )

    assert restore_plane_checkpoint(
        target,
        checkpoint,
        run_spec=target_spec,
    ) == 1
    _assert_state_equal(source, target)


def test_m03_prescribed_lift_change_rejects_before_target_mutation(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path, "run")
    source = _runtime(spec)
    target = _runtime(spec)
    source.advance(1)
    checkpoint = capture_plane_checkpoint(source, run_spec=spec)
    checkpoint = _checkpoint_with_lifting(
        checkpoint,
        spec,
        _synthetic_lifting("a" * 64, "same_initializer"),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "plane_lifting_restart_metadata",
        lambda _adapter: _synthetic_lifting(
            "b" * 64,
            "same_initializer",
        ),
    )
    before = _state(target)

    with pytest.raises(ValueError, match="static lifting identity"):
        restore_plane_checkpoint(target, checkpoint, run_spec=spec)

    assert all(
        torch.equal(value, target.fields[name])
        for name, value in before.items()
    )
    assert target.completed_steps == 0


def test_forward_dynamics_change_rejects_before_target_mutation(tmp_path):
    source_spec = _spec(tmp_path, "source", dt=0.005)
    target_spec = _spec(tmp_path, "target", dt=0.004)
    source = _runtime(source_spec)
    target = _runtime(target_spec)
    source.advance(1)
    checkpoint = capture_plane_checkpoint(source, run_spec=source_spec)
    before = _state(target)

    with pytest.raises(ValueError, match="compatibility identity"):
        restore_plane_checkpoint(target, checkpoint, run_spec=target_spec)

    assert all(
        torch.equal(value, target.fields[name])
        for name, value in before.items()
    )
    assert target.completed_steps == 0


def test_m04_rc1_lifted_legacy_checkpoint_rejects_exact_upgrade(tmp_path):
    spec = _spec(tmp_path, "legacy", nx=7, ny=7)
    runtime = _runtime(spec)
    checkpoint = _legacy_checkpoint(runtime, spec)
    lifted = replace(
        checkpoint,
        lifting_restart={"representation": "legacy_nonhomogeneous_q"},
    )

    with pytest.raises(ValueError, match="rc1 lifted"):
        upgrade_plane_checkpoint_v1(
            lifted,
            source_run_spec=spec,
            source_release_generation=SourceReleaseGeneration.RC1,
        )


def test_legacy_v1_cannot_restore_without_explicit_upgrade(tmp_path):
    spec = _spec(tmp_path, "legacy", nx=7, ny=7)
    source = _runtime(spec)
    target = _runtime(spec)
    legacy = _legacy_checkpoint(source, spec)

    with pytest.raises(ValueError, match="explicit identity upgrade"):
        restore_plane_checkpoint(target, legacy, run_spec=spec)


@pytest.mark.parametrize("generation", tuple(SourceReleaseGeneration))
def test_m05_even_grid_legacy_checkpoint_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
    generation,
):
    spec = _spec(tmp_path, generation.value, nx=8, ny=7)
    runtime = _runtime(spec)
    legacy = _legacy_checkpoint(runtime, spec)
    source = write_plane_checkpoint(tmp_path / "legacy", legacy)
    target = tmp_path / "upgraded"
    monkeypatch.setattr(
        checkpoint_module,
        "_load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(ValueError, match="pre-RC4.1 Nyquist dynamics"):
        upgrade_plane_checkpoint_directory_v1(
            source,
            target,
            source_run_spec=spec,
            source_release_generation=generation,
        )

    assert not target.exists()


@pytest.mark.parametrize("generation", tuple(SourceReleaseGeneration))
def test_m06_odd_grid_legacy_checkpoint_upgrades_without_source_mutation(
    tmp_path,
    generation,
):
    spec = _spec(tmp_path, generation.value, nx=7, ny=7)
    source_runtime = _runtime(spec)
    target_runtime = _runtime(spec)
    source_runtime.advance(2)
    legacy = _legacy_checkpoint(source_runtime, spec)
    source = write_plane_checkpoint(
        tmp_path / f"legacy_{generation.value}",
        legacy,
    )
    before = _tree_sha256(source)
    upgraded_path = upgrade_plane_checkpoint_directory_v1(
        source,
        tmp_path / f"upgraded_{generation.value}",
        source_run_spec=spec,
        source_release_generation=generation,
    )

    assert _tree_sha256(source) == before
    assert read_plane_checkpoint_header(source).format_version == (
        PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )
    upgraded = load_plane_checkpoint(upgraded_path)
    assert upgraded.format_version == PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    assert upgraded.migration_provenance[
        "rc4_1_nyquist_applicability_basis"
    ] == "both_periodic_axis_lengths_are_odd"
    assert restore_plane_checkpoint(
        target_runtime,
        upgraded,
        run_spec=spec,
    ) == 2
    _assert_state_equal(source_runtime, target_runtime)


def test_legacy_opaque_identity_mismatch_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
):
    source_spec = _spec(tmp_path, "source", nx=7, ny=7, dt=0.005)
    wrong_spec = _spec(tmp_path, "wrong", nx=7, ny=7, dt=0.004)
    runtime = _runtime(source_spec)
    legacy = _legacy_checkpoint(runtime, source_spec)
    source = write_plane_checkpoint(tmp_path / "legacy", legacy)
    monkeypatch.setattr(
        checkpoint_module,
        "_load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(ValueError, match="opaque identity"):
        upgrade_plane_checkpoint_directory_v1(
            source,
            tmp_path / "upgraded",
            source_run_spec=wrong_spec,
            source_release_generation=SourceReleaseGeneration.RC3,
        )


def test_legacy_upgrade_target_cannot_mutate_source_tree(tmp_path):
    spec = _spec(tmp_path, "source", nx=7, ny=7)
    legacy = _legacy_checkpoint(_runtime(spec), spec)
    source = write_plane_checkpoint(tmp_path / "legacy", legacy)
    before = _tree_sha256(source)

    with pytest.raises(ValueError, match="outside the source"):
        upgrade_plane_checkpoint_directory_v1(
            source,
            source / "upgraded",
            source_run_spec=spec,
            source_release_generation=SourceReleaseGeneration.RC3,
        )

    assert _tree_sha256(source) == before


def test_v2_header_rejects_identity_digest_tamper_without_tensor_load(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path, "source")
    checkpoint = capture_plane_checkpoint(_runtime(spec), run_spec=spec)
    directory = write_plane_checkpoint(tmp_path / "checkpoint", checkpoint)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["compatibility_identity_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(ValueError, match="identity checksum mismatch"):
        load_plane_checkpoint(directory)


def test_unknown_checkpoint_version_rejects_without_tensor_load(
    tmp_path,
    monkeypatch,
):
    spec = _spec(tmp_path, "source")
    checkpoint = capture_plane_checkpoint(_runtime(spec), run_spec=spec)
    directory = write_plane_checkpoint(tmp_path / "checkpoint", checkpoint)
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 99
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(ValueError, match="unsupported Plane"):
        load_plane_checkpoint(directory)
