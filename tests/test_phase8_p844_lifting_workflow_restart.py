"""P8.4.4 Plane static-lifting runtime, workflow, and restart tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    SpectralSolver,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    beris_edwards_bulk_molecular_field_components,
    strong_homeotropic_q,
)
from pssolver.operators.projection import BasisAwareSpectralProjector
from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p844_lifting_workflow_restart.json"
)














def test_lifted_stress_projects_nonzero_wall_molecular_field_in_dct_basis():
    errors = []
    dirichlet_errors = []
    height = 10.0
    scalar_order = 0.4
    for nz in (16, 32, 64):
        solver = SpectralSolver(
            (4, 4, nz),
            L=(4.0, 4.0, height),
            device="cpu",
            dtype=torch.float64,
            spectral_storage="full_complex",
        )
        projector = BasisAwareSpectralProjector(
            solver,
            rule="cubic_half",
            transform_execution="full",
        )
        z = (torch.arange(nz, dtype=torch.float64) + 0.5) * height / nz
        theta = 0.5 * torch.pi * z / height
        zero = torch.zeros_like(z)
        q = torch.stack(
            (
                scalar_order * (torch.sin(theta).square() - 1.0 / 3.0),
                zero,
                scalar_order * torch.sin(theta) * torch.cos(theta),
                torch.full_like(z, -scalar_order / 3.0),
                zero,
            )
        )[:, None, None, :].expand(-1, 4, 4, -1)
        raw_h = torch.stack(
            beris_edwards_bulk_molecular_field_components(
                tuple(q),
                ldg_a=-0.1,
                ldg_b=-0.3,
                ldg_c=0.3,
            )
        )
        neumann = ("periodic", "periodic", "neumann")
        dirichlet = ("periodic", "periodic", "dirichlet")
        projected = projector.inverse_transform(
            projector.forward_transform(raw_h, neumann),
            neumann,
        )
        incorrectly_projected = projector.inverse_transform(
            projector.forward_transform(raw_h, dirichlet),
            dirichlet,
        )
        errors.append(float((projected - raw_h).abs().max().item()))
        dirichlet_errors.append(
            float((incorrectly_projected - raw_h).abs().max().item())
        )

    assert errors[1] < 0.5 * errors[0]
    assert errors[2] < 0.5 * errors[1]
    assert dirichlet_errors[-1] > 100.0 * errors[-1]














def _public_simulation(tmp_path: Path) -> Simulation:
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
    geometry = PlaneSlab(shape=(8, 8, 6), lengths=(100.0, 100.0, 20.0))
    policy = strong_homeotropic_q(
        scalar_order=0.6,
        face_normals={
            (2, "lower"): (0.0, 0.0, -1.0),
            (2, "upper"): (0.0, 0.0, 1.0),
        },
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.005, refresh={"mode": "disabled"}),
        initial_condition=GeneratedInitialCondition(
            "extruded_defect_gas",
            parameters={
                "seed": 24,
                "num_defect_pairs": 1,
                "defect_min_separation": 10.0,
                "defect_core_radius": 1.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
                "initial_s": 1.0 / 3.0,
            },
        ),
        execution=TorchSpectralExecution(
            runtime_path="legacy_production",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / "public_lifted",
            steps=1,
            save_start_step=0,
            save_interval=1,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
            checkpoint_interval=1,
        ),
    )


def test_public_runner_rejects_historical_lifted_capability_before_output(
    tmp_path,
):
    simulation = _public_simulation(tmp_path)
    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)

    assert caught.value.code is (
        PublicCompilationRejectionCode
        .UNSUPPORTED_LIFTED_WALL_MOLECULAR_FIELD
    )
    assert not (tmp_path / "public_lifted").exists()


def test_p844_record_matches_qualified_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    s16_successor = json.loads(
        (
            ROOT
            / "notes/PSSolver_v0_3_0_s03_successor_72_s16_r0043_rework.json"
        ).read_text(encoding="utf-8")
    )["historical_successor_sources"]
    assert record["classification"] == (
        "PASS_P8_4_4_PLANE_STATIC_LIFTING_WORKFLOW_RESTART"
    )
    assert record["authorization"]["p8_4_4_complete"] is True
    assert record["authorization"]["p8_4_5_performed"] is False
    assert record["nonhomogeneous_neumann_supported"] is False
    assert record["representation"]["evolved_field"] == (
        "homogeneous_remainder"
    )
    assert record["output"]["saved_q"] == "physical_field"
    # Later qualification and compatibility-maintenance slices changed these
    # sources. Keep the P8.4.4 hashes as historical evidence instead of
    # rewriting the already qualified record.
    superseded_after_p844 = {
        "pssolver/api/runner.py",
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/configuration/public_simulation_runner.py",
        "pssolver/models/active_nematics/beris_edwards.py",
        "pssolver/models/active_nematics/stokes.py",
        "pssolver/operators/lifting.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/runtime/static_lifting.py",
        "pssolver/workflows/plane_checkpoint.py",
        "pssolver/workflows/plane_observation.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_after_p844:
            continue
        if relative in s16_successor:
            identity = s16_successor[relative]
            assert identity["historical_sha256"] == expected, relative
            assert (
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                == resolve_s03_successor(relative, identity["current_sha256"])
            ), relative
            continue
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
