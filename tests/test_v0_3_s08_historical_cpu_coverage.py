"""S-08 authentic historical fixtures for A2-25 and N5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.functional import (
    FunctionalCheckpointCompatibilityError,
    build_channel_activity_functional_runtime,
    channel_activity_functional_request,
)
from pssolver.geometries import RectangularChannel
from pssolver.io.checkpoint_identity import SourceReleaseGeneration
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.workflows import plane_checkpoint as plane_checkpoint_module
from pssolver.workflows.plane_checkpoint import (
    upgrade_plane_checkpoint_directory_v1,
)


FIXTURES = Path(__file__).parent / "fixtures" / "historical_cpu_coverage_v1"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tree_sha256(directory: Path) -> str:
    encoded = json.dumps(
        _tree(directory), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _channel_runtime(tmp_path: Path, fixture: Path):
    shape = (7, 6, 6)
    model = CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )
    geometry = RectangularChannel(shape=shape, lengths=(7.0, 6.0, 6.0))
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": no_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(fixture / "snapshot", step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / "unused_output",
            steps=2,
            save_start_step=0,
            save_interval=2,
            diagnostic_interval=2,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
    )
    return build_channel_activity_functional_runtime(
        channel_activity_functional_request(simulation.specification)
    )


def _plane_source_spec(tmp_path: Path):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused_output",
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
        steps=2,
        save_start_step=0,
        save_interval=2,
        diagnostic_interval=2,
        spectral_refresh_steps=3,
        seed=24,
        defect_min_separation=2.0,
    )


def test_s08_fixture_manifest_binds_exact_release_outputs():
    assert MANIFEST["fixture_schema_version"] == 1
    assert set(MANIFEST["cases"]) == {
        "rc1_channel_functional_odd",
        "rc2_channel_functional_odd",
        "rc1_plane_lifted",
    }
    assert _sha256(FIXTURES / "generate_fixtures.py") == MANIFEST[
        "generator"
    ]["sha256"]
    for name, record in MANIFEST["cases"].items():
        directory = FIXTURES / name
        assert len(_tree(directory)) == record["file_count"]
        assert _tree_sha256(directory) == record["tree_sha256"]
        generation = json.loads(
            (directory / "generation.json").read_text(encoding="utf-8")
        )
        release = MANIFEST["released_sources"][record["source_release"]]
        assert generation["source"]["commit"] == release["commit"]
        assert generation["source"]["pssolver_version"] == release["version"]


@pytest.mark.parametrize(
    ("generation", "migration_kind", "trajectory_equivalent", "continuation_exact"),
    (
        ("rc1", "state_only_migration", False, False),
        ("rc2", "exact_legacy_upgrade", True, True),
    ),
)
def test_authentic_odd_channel_functional_checkpoint_has_honest_disposition(
    tmp_path,
    generation,
    migration_kind,
    trajectory_equivalent,
    continuation_exact,
):
    fixture = FIXTURES / f"{generation}_channel_functional_odd"
    runtime = _channel_runtime(tmp_path / generation, fixture)
    source = fixture / "checkpoint"
    before = _tree(source)
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="requires explicit migration",
    ):
        runtime.checkpoint_bridge.import_checkpoint(source)

    target = runtime.checkpoint_bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / f"{generation}_upgraded",
        source_release_generation=generation,
    )
    imported = runtime.checkpoint_bridge.import_checkpoint(target)
    metadata = json.loads((target / "checkpoint.json").read_text())
    provenance = metadata["migration_provenance"]
    assert provenance["migration_kind"] == migration_kind
    assert provenance["trajectory_equivalent"] is trajectory_equivalent
    assert provenance["periodic_axis_size"] == 7
    assert _tree(source) == before

    controls = {
        "activity": torch.full(
            runtime.control_specs[0].tensor.shape,
            0.01,
            dtype=torch.float64,
        )
    }
    continued = runtime.step(imported.state, controls, 1)
    oracle = fixture / "continuation_oracle"
    expected = (
        torch.from_numpy(np.load(oracle / "q_physical_step_2.npy")),
        torch.from_numpy(np.load(oracle / "q_spectral_step_2.npy")),
    )
    observed_exact = all(
        torch.equal(left, right)
        for left, right in zip(continued, expected, strict=True)
    )
    assert observed_exact is continuation_exact


def test_authentic_rc1_lifted_plane_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
):
    fixture = FIXTURES / "rc1_plane_lifted"
    source = fixture / "checkpoint"
    before = _tree(source)
    generation = json.loads((fixture / "generation.json").read_text())
    spec = _plane_source_spec(tmp_path)
    assert spec.runtime_identity_sha256() == generation[
        "runtime_identity_sha256"
    ]
    monkeypatch.setattr(
        plane_checkpoint_module,
        "_load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor payload load started"),
    )
    target = tmp_path / "forbidden_upgrade"
    with pytest.raises(ValueError, match="rc1 lifted Plane checkpoint"):
        upgrade_plane_checkpoint_directory_v1(
            source,
            target,
            source_run_spec=spec,
            source_release_generation=SourceReleaseGeneration.RC1,
        )
    assert not target.exists()
    assert _tree(source) == before
