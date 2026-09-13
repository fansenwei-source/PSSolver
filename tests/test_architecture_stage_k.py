"""Qualification tests for the Stage K Plane shadow-run boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import pssolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.execution import IncompressibleStokesSystemSpec
from pssolver.experimental import (
    ExperimentalPlaneShadowRun,
    PlaneBerisEdwardsSolverOptions,
    build_experimental_model_runtime,
    capture_shadow_checkpoint,
    compare_shadow_to_production_metadata,
    create_beris_edwards_plane_geometry_solver_registry,
    load_shadow_checkpoint,
    plane_beris_edwards_shadow_signature,
    restore_shadow_checkpoint,
    write_shadow_checkpoint,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    NEMATIC_FORCE_COMPONENTS,
    Q_COMPONENTS,
    VELOCITY_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsPlaneCoupledModel,
)
from pssolver.snapshots import load_snapshot


P = PeriodicBC()
N = HomogeneousNeumannBC()
D = HomogeneousDirichletBC()
EVEN = BoundarySet((P, P, N))
ODD = BoundarySet((P, P, D))
SHAPE = (10, 8, 7)
LENGTHS = (5.0, 4.0, 3.5)
DT = 0.005


def _model(*, active_prefactor=-0.18):
    constitutive = BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=EVEN,
        tangential_boundaries=EVEN,
        normal_boundaries=ODD,
        pressure_boundaries=EVEN,
        parameters=BerisEdwardsConstitutiveParameters(
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
            ldg_l1=0.04,
            flow_alignment=0.31,
            active_prefactor=active_prefactor,
        ),
        stokes_system=IncompressibleStokesSystemSpec(
            name="flow",
            force_components=NEMATIC_FORCE_COMPONENTS,
            velocity_components=VELOCITY_COMPONENTS,
            pressure_component="p",
            viscosity=2.0 / 3.0,
        ),
        initial_amplitude=0.02,
    )
    return BerisEdwardsPlaneCoupledModel(
        constitutive_model=constitutive,
        rotational_viscosity=2.94,
    )


def _numerics(*, half_spectrum):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.TRUNCATED,
        spectral_storage=(
            SpectralStorage.HERMITIAN_HALF
            if half_spectrum
            else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=1 if half_spectrum else None,
    )


def _build(*, half_spectrum=False, dt=DT, active_prefactor=-0.18):
    runtime = build_experimental_model_runtime(
        _model(active_prefactor=active_prefactor),
        PlaneSlab(DomainSpec(SHAPE, LENGTHS)),
        _numerics(half_spectrum=half_spectrum),
        dt=dt,
        device="cpu",
        batch_size=1,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space="spectral",
                    stress_divergence_sum_space="spectral",
                    pointwise_execution="eager",
                )
            )
        ),
    )
    runtime.solver.integrator.set_spectral_refresh_interval(40)
    return runtime


def _initial_values():
    generator = torch.Generator().manual_seed(2468)
    return {
        name: 0.04
        * torch.randn(
            SHAPE,
            dtype=torch.float64,
            generator=generator,
        )
        for name in Q_COMPONENTS
    }


def _production_metadata(runtime):
    signature = plane_beris_edwards_shadow_signature(runtime)
    parameters = signature["model"]["parameters"]
    return {
        "solver": dict(signature["solver"]),
        "model": {
            "name": "active_nematics",
            "variant": "beris_edwards_complete_nematic_stress_stokes",
            "Q_convention": signature["model"]["Q_convention"],
            "parameters": {
                "alpha": -parameters["active_stress_prefactor"],
                "beta": -1.0,
                "eta": parameters["eta"],
                "flow_alignment_lambda": parameters[
                    "flow_alignment_lambda"
                ],
                "fric": parameters["fric"],
                "ldg_A": parameters["ldg_A"],
                "ldg_B": parameters["ldg_B"],
                "ldg_C": parameters["ldg_C"],
                "ldg_L1": parameters["ldg_L1"],
                "rotational_viscosity_gamma": parameters[
                    "rotational_viscosity_gamma"
                ],
            },
        },
        "boundary_conditions": dict(signature["boundary_conditions"]),
        "numerics": {
            "dealiasing": {"rule": signature["numerics"]["dealias_rule"]},
            "velocity_zero_mode": signature["numerics"][
                "velocity_zero_mode"
            ],
            "molecular_field_linear_space": signature["numerics"][
                "molecular_field_linear_space"
            ],
            "stress_divergence_sum_space": signature["numerics"][
                "stress_divergence_sum_space"
            ],
            "spectral_refresh": {
                "effective_interval_steps": signature["numerics"][
                    "spectral_refresh_interval_steps"
                ]
            },
            "pointwise_kernels": {
                "effective": signature["numerics"]["pointwise_execution"]
            },
            "transforms": {
                "hermitian_axis": signature["numerics"]["hermitian_axis"],
                "projected_transform_execution": signature["numerics"][
                    "projected_transform_execution"
                ],
            },
        },
    }


def _assert_runtime_states_close(left, right, *, exact):
    left.synchronize_algebraic_for_observation()
    right.synchronize_algebraic_for_observation()
    for name in (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p"):
        if exact:
            assert torch.equal(
                left.solver.fields[name],
                right.solver.fields[name],
            )
            assert torch.equal(
                left.solver.fields[f"{name}.hat"],
                right.solver.fields[f"{name}.hat"],
            )
        else:
            torch.testing.assert_close(
                left.solver.fields[name],
                right.solver.fields[name],
                rtol=2.0e-12,
                atol=2.0e-13,
            )


def test_shadow_metadata_comparison_covers_migrated_production_contract():
    runtime = _build(half_spectrum=True)
    production = _production_metadata(runtime)
    comparison = compare_shadow_to_production_metadata(runtime, production)

    assert comparison.compatible
    assert comparison.differing_paths == ()
    comparison.require_compatible()
    json.dumps(comparison.to_metadata(), allow_nan=False, sort_keys=True)

    production["model"]["parameters"]["ldg_L1"] = 0.05
    mismatch = compare_shadow_to_production_metadata(runtime, production)
    assert not mismatch.compatible
    assert mismatch.differing_paths == ("model.parameters.ldg_L1",)
    with pytest.raises(ValueError, match="model.parameters.ldg_L1"):
        mismatch.require_compatible()

    production = _production_metadata(runtime)
    production["numerics"]["spectral_refresh"][
        "effective_interval_steps"
    ] = 41
    refresh_mismatch = compare_shadow_to_production_metadata(
        runtime,
        production,
    )
    assert refresh_mismatch.differing_paths == (
        "numerics.spectral_refresh_interval_steps",
    )


def test_shadow_signature_matches_actual_production_dry_run(tmp_path):
    command = (
        sys.executable,
        str(Path(__file__).parents[1] / "Plane_beris_edwards_stokes.py"),
        "--activity-number",
        "10.5",
        "--output-dir",
        str(tmp_path / "unused_dry_run_output"),
        "--height",
        "3.5",
        "--parameterization",
        "fixed-k",
        "--frank-k",
        "0.02",
        "--lx",
        "5.0",
        "--ly",
        "4.0",
        "--nx",
        "10",
        "--ny",
        "8",
        "--nz",
        "7",
        "--dt",
        "0.005",
        "--steps",
        "1",
        "--save-start-step",
        "0",
        "--ldg-a",
        "0.0",
        "--ldg-b",
        "-0.3",
        "--ldg-c",
        "0.3",
        "--gamma",
        "2.94",
        "--flow-alignment",
        "0.31",
        "--eta",
        str(2.0 / 3.0),
        "--beta",
        "-1.0",
        "--dtype",
        "float64",
        "--dealias-rule",
        "cubic_half",
        "--projected-transform-execution",
        "truncated",
        "--molecular-field-linear-space",
        "spectral",
        "--stress-divergence-sum-space",
        "spectral",
        "--pointwise-execution",
        "eager",
        "--transform-execution-order",
        "real_first",
        "--spectral-storage",
        "hermitian_half",
        "--zero-mode-policy",
        "zero_mean",
        "--dry-run",
    )
    result = subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    production_metadata = json.loads(result.stdout)
    comparison = compare_shadow_to_production_metadata(
        _build(half_spectrum=True),
        production_metadata,
    )

    comparison.require_compatible()
    assert comparison.compatible
    assert not (tmp_path / "unused_dry_run_output").exists()


def test_shadow_run_injects_initial_q_and_writes_production_layout(tmp_path):
    runtime = _build(half_spectrum=True)
    reference = _build(half_spectrum=True)
    initial = _initial_values()
    reference.reset(initial)

    run = ExperimentalPlaneShadowRun(
        runtime,
        tmp_path / "shadow",
        initial_values=initial,
        initial_condition_metadata={
            "name": "stage_k_external_q",
            "source": "test",
            "S_initial": 0.4,
        },
    )
    _assert_runtime_states_close(runtime, reference, exact=True)

    running_metadata = json.loads(
        (run.output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert running_metadata["status"] == "running"
    assert running_metadata["initial_condition"]["name"] == (
        "stage_k_external_q"
    )
    assert len(running_metadata["initial_condition"]["raw_q_sha256"]) == 64
    assert len(
        running_metadata["initial_condition"]["projected_q_sha256"]
    ) == 64
    assert running_metadata["checkpointing"] == {
        "complete_checkpoint_format_version": 1,
        "observation_layout": "Q/u/p production filenames",
        "transient_fields_checkpointed": False,
        "stored_algebraic_fields_checkpointed": False,
        "stored_algebraic_fields_reconstructed": True,
    }

    assert run.advance(3) == 3
    observation = run.save_observation()
    assert observation.q.shape == (*SHAPE, 5)
    assert observation.velocity.shape == (*SHAPE, 3)
    assert observation.pressure.shape == SHAPE
    assert observation.q.dtype == np.float64
    assert observation.velocity.dtype == np.float64
    assert observation.pressure.dtype == np.float64
    for prefix in ("Q", "u", "p"):
        assert (run.output_directory / f"{prefix}_3.npy").is_file()

    loaded = load_snapshot(
        run.output_directory,
        3,
        expected_shape=SHAPE,
        expected_S_bulk=1.0 / 3.0,
        require_S_initial=True,
    )
    assert loaded.step == 3
    final = run.complete()
    assert final.step == 3
    assert (run.output_directory / "COMPLETE").read_text() == "complete\n"
    final_metadata = json.loads(
        (run.output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert final_metadata["status"] == "complete"
    assert final_metadata["completed_steps"] == 3
    with pytest.raises(RuntimeError, match="already complete"):
        run.advance(1)


def test_shadow_run_rejects_partial_or_precision_changing_initial_state(
    tmp_path,
):
    initial = _initial_values()
    partial = dict(initial)
    partial.pop("Qyz")
    with pytest.raises(ValueError, match="declared order"):
        ExperimentalPlaneShadowRun(
            _build(),
            tmp_path / "partial",
            initial_values=partial,
            initial_condition_metadata={"name": "partial"},
        )

    wrong_precision = dict(initial)
    wrong_precision["Qxx"] = wrong_precision["Qxx"].float()
    with pytest.raises(ValueError, match="expected torch.float64"):
        ExperimentalPlaneShadowRun(
            _build(),
            tmp_path / "wrong_precision",
            initial_values=wrong_precision,
            initial_condition_metadata={"name": "wrong_precision"},
        )


def test_shadow_run_refuses_nonempty_output_before_mutating_runtime(tmp_path):
    runtime = _build()
    spatial_before = runtime.solver.fields.spatial.clone()
    spectral_before = runtime.solver.fields.spectral.clone()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user-data.txt").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="nonempty directory"):
        ExperimentalPlaneShadowRun(
            runtime,
            output,
            initial_values=_initial_values(),
            initial_condition_metadata={"name": "must_not_start"},
        )

    assert runtime.solver.integrator.step_count == 0
    assert runtime.solver.integrator.refresh_count == 0
    assert torch.equal(runtime.solver.fields.spatial, spatial_before)
    assert torch.equal(runtime.solver.fields.spectral, spectral_before)
    assert (output / "user-data.txt").read_text(encoding="utf-8") == (
        "preserve\n"
    )


@pytest.mark.parametrize("half_spectrum", (False, True))
def test_checkpoint_restart_matches_continuous_trajectory(
    tmp_path,
    half_spectrum,
):
    initial = _initial_values()
    continuous = _build(half_spectrum=half_spectrum)
    segment = _build(half_spectrum=half_spectrum)
    resumed = _build(half_spectrum=half_spectrum)
    for runtime in (continuous, segment, resumed):
        runtime.solver.integrator.set_spectral_refresh_interval(3)
        runtime.reset(initial)

    continuous.solver.run(8)
    segment.solver.run(4)
    checkpoint = capture_shadow_checkpoint(segment)
    assert checkpoint.completed_steps == 4
    assert checkpoint.integrator_refresh_count == 1
    assert checkpoint.integrator_step_count == 1
    path = write_shadow_checkpoint(tmp_path / "checkpoint", checkpoint)
    loaded = load_shadow_checkpoint(path)

    assert restore_shadow_checkpoint(resumed, loaded) == 4
    resumed.solver.run(4)
    assert resumed.solver.integrator.refresh_count == 2
    assert resumed.solver.integrator.step_count == 2
    _assert_runtime_states_close(continuous, resumed, exact=True)


def test_checkpoint_checksum_and_runtime_identity_are_hard_gates(tmp_path):
    source = _build()
    source.solver.run(2)
    path = write_shadow_checkpoint(
        tmp_path / "checkpoint",
        capture_shadow_checkpoint(source),
    )

    incompatible = _build(dt=0.006)
    with pytest.raises(ValueError, match="runtime identity"):
        restore_shadow_checkpoint(
            incompatible,
            load_shadow_checkpoint(path),
        )

    tensor_path = path / "evolved_spatial__Qxx.npy"
    contents = bytearray(tensor_path.read_bytes())
    contents[-1] ^= 1
    tensor_path.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_shadow_checkpoint(path)


def test_shadow_runner_restart_records_source_and_continues_numbering(tmp_path):
    initial = _initial_values()
    source = _build()
    source.solver.integrator.set_spectral_refresh_interval(3)
    source.reset(initial)
    source.solver.run(4)
    checkpoint = capture_shadow_checkpoint(source)

    target = _build()
    target.solver.integrator.set_spectral_refresh_interval(3)
    run = ExperimentalPlaneShadowRun.from_checkpoint(
        target,
        tmp_path / "resumed",
        checkpoint,
    )
    assert run.completed_steps == 4
    assert run.advance(2) == 6
    final = run.complete()
    assert final.step == 6
    assert (run.output_directory / "Q_6.npy").is_file()
    metadata = json.loads(
        (run.output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["restart"]["source_completed_steps"] == 4
    assert metadata["completed_steps"] == 6


def test_stage_k_remains_opt_in_and_production_drivers_are_unchanged():
    assert not hasattr(pssolver, "ExperimentalPlaneShadowRun")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/solver.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "ExperimentalPlaneShadowRun" not in text
        assert "ShadowRunCheckpoint" not in text
