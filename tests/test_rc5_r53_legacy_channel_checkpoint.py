"""R5-3 regression tests for released P7 legacy Channel checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import (
    CHANNEL_FORWARD_STOKES_NYQUIST_VERSION,
    ChannelActiveNematicRunSpec,
)
from pssolver.runtime import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.workflows import (
    CHANNEL_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    load_channel_checkpoint,
    restore_channel_checkpoint,
)


FIXTURES = Path(__file__).parent / "fixtures" / "legacy_channel_checkpoints_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(directory: Path) -> tuple[int, str]:
    files = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    payload = json.dumps(
        files,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return len(files), hashlib.sha256(payload).hexdigest()


def _spec(periodic_length: int) -> ChannelActiveNematicRunSpec:
    wall_length = 5 if periodic_length == 7 else 6
    return ChannelActiveNematicRunSpec(
        shape=(periodic_length, wall_length, 5),
        lengths=(float(periodic_length), float(wall_length), 5.0),
        dt=1.0e-3,
        steps=2,
        save_interval=2,
        diagnostic_interval=2,
        device="cpu",
        dtype="float32",
        activity=0.2,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=100,
        pressure_fixed_iterations=8,
        runtime_path="legacy_channel",
        seed=24,
    )


def _runtime(spec: ChannelActiveNematicRunSpec):
    values = {
        name: torch.zeros(spec.shape, dtype=torch.float32)
        for name in Q_COMPONENTS
    }
    metadata = {
        "configuration": spec.identity_metadata(),
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    request = ChannelRuntimeBuildRequest(spec, metadata, values, "cpu")

    def builder():
        material = spec.components.material
        pressure = spec.components.pressure_solver
        solver = build_active_nematic_channel(
            spec.shape,
            spec.lengths,
            spec.dt,
            values,
            device="cpu",
            batchsize=spec.batch_size,
            rho=material.rho,
            elastic_constant=material.elastic_constant,
            beta=material.beta,
            friction=material.friction,
            viscosity=material.viscosity,
            pressure_rel_tol=pressure.relative_tolerance,
            pressure_max_iter=pressure.max_iterations,
            pressure_fixed_iterations=pressure.fixed_iterations,
        )
        solver.parameters["alpha"] = torch.full(
            (spec.batch_size, *spec.shape),
            material.activity,
            dtype=solver.dtype,
            device=solver.device,
        )
        return solver

    return build_channel_active_nematic_runtime(
        request,
        legacy_builder=builder,
    )


def _state(adapter) -> dict[str, object]:
    return {
        "spatial": adapter.fields.spatial.detach().clone(),
        "spectral": adapter.fields.spectral.detach().clone(),
        "pressure": adapter.capture_pressure_guess(),
        "completed_steps": adapter.completed_steps,
        "integrator_step_count": int(adapter.solver.integrator.step_count),
        "integrator_refresh_count": int(adapter.solver.integrator.refresh_count),
    }


def _assert_state(adapter, expected: dict[str, object]) -> None:
    assert torch.equal(adapter.fields.spatial, expected["spatial"])
    assert torch.equal(adapter.fields.spectral, expected["spectral"])
    assert torch.equal(adapter.capture_pressure_guess(), expected["pressure"])
    assert adapter.completed_steps == expected["completed_steps"]
    assert int(adapter.solver.integrator.step_count) == expected[
        "integrator_step_count"
    ]
    assert int(adapter.solver.integrator.refresh_count) == expected[
        "integrator_refresh_count"
    ]


def _assert_checkpoint_restored(adapter, checkpoint) -> None:
    for name in Q_COMPONENTS:
        assert torch.equal(adapter.fields[name], checkpoint.evolved_spatial[name])
        assert torch.equal(
            adapter.fields[f"{name}.hat"],
            checkpoint.evolved_spectral[name],
        )
    for name in ("ux", "uy", "uz", "p"):
        assert torch.equal(
            adapter.fields[name],
            checkpoint.algebraic_spatial[name],
        )
        assert torch.equal(
            adapter.fields[f"{name}.hat"],
            checkpoint.algebraic_spectral[name],
        )
    assert torch.equal(
        adapter.capture_pressure_guess(),
        checkpoint.pressure_guess,
    )


def test_released_fixture_manifest_is_provenance_bound():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["fixture_schema_version"] == 1
    assert set(manifest["cases"]) == {
        "rc1_even",
        "rc1_odd",
        "rc2_even",
        "rc3_even",
    }
    for name, record in manifest["cases"].items():
        directory = FIXTURES / name
        assert _sha256(directory / "checkpoint.json") == record[
            "checkpoint_sha256"
        ]
        assert _tree_sha256(directory) == (
            record["file_count"],
            record["tree_sha256"],
        )


def test_current_identity_versions_forward_stokes_nyquist_semantics():
    for periodic_length in (7, 8):
        spec = _spec(periodic_length)
        assert spec.runtime_identity_metadata()["dynamics_versions"] == {
            "forward_stokes_nyquist": (
                CHANNEL_FORWARD_STOKES_NYQUIST_VERSION
            )
        }
        metadata = json.loads(
            (
                FIXTURES
                / ("rc1_odd" if periodic_length == 7 else "rc1_even")
                / "checkpoint.json"
            ).read_text()
        )
        assert spec.pre_rc5_runtime_identity_sha256() == metadata[
            "runtime_identity_sha256"
        ]
        assert spec.runtime_identity_sha256() != metadata[
            "runtime_identity_sha256"
        ]


def test_real_rc1_even_checkpoint_rejects_before_target_mutation():
    spec = _spec(8)
    target = _runtime(spec)
    checkpoint = load_channel_checkpoint(FIXTURES / "rc1_even")
    before = _state(target)

    assert checkpoint.format_version == (
        CHANNEL_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )
    with pytest.raises(
        ValueError,
        match="pre-RC4.1 forward Nyquist dynamics",
    ):
        restore_channel_checkpoint(
            target,
            checkpoint,
            runtime_identity_sha256=spec.runtime_identity_sha256(),
            pre_rc5_runtime_identity_sha256=(
                spec.pre_rc5_runtime_identity_sha256()
            ),
        )
    _assert_state(target, before)


@pytest.mark.parametrize("case", ("rc1_odd", "rc2_even", "rc3_even"))
def test_compatible_released_checkpoint_restores_exact_state(case):
    record = json.loads((FIXTURES / "manifest.json").read_text())[
        "cases"
    ][case]
    spec = _spec(record["periodic_axis_length"])
    target = _runtime(spec)
    checkpoint = load_channel_checkpoint(FIXTURES / case)

    assert restore_channel_checkpoint(
        target,
        checkpoint,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
        pre_rc5_runtime_identity_sha256=(
            spec.pre_rc5_runtime_identity_sha256()
        ),
    ) == checkpoint.completed_steps
    _assert_checkpoint_restored(target, checkpoint)
