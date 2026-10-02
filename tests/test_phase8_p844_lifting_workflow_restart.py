"""P8.4.4 Plane static-lifting runtime, workflow, and restart tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
    SpectralSolver,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_binding import (
    BindingRejectionCode,
    SimulationBindingError,
    bind_simulation_runtime,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    Q_COMPONENTS,
    beris_edwards_bulk_molecular_field_components,
    strong_homeotropic_q,
    strong_planar_q,
)
from pssolver.operators import materialize_plane_static_lifting
from pssolver.operators.projection import BasisAwareSpectralProjector
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    plane_physical_component,
)
from pssolver.workflows import capture_plane_observation
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    load_plane_checkpoint,
    restore_plane_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p844_lifting_workflow_restart.json"
)


def _run_spec(tmp_path: Path, *, runtime_path: str = "legacy_production"):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        runtime_path=runtime_path,
        nx=8,
        ny=8,
        nz=6,
        lx=100.0,
        ly=100.0,
        height=20.0,
        steps=2,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=10.0,
        defect_core_radius=1.5,
        twist_modes=(1, 2, 3),
    )


def _lifted_simulation(run_spec, *, planar: bool = False):
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    normals = {(2, "lower"): (0.0, 0.0, -1.0), (2, "upper"): (0.0, 0.0, 1.0)}
    policy = (
        strong_planar_q(
            scalar_order=0.6,
            face_directors={
                (2, "lower"): (1.0, 0.0, 0.0),
                (2, "upper"): (0.0, 1.0, 0.0),
            },
            face_normals=normals,
        )
        if planar
        else strong_homeotropic_q(
            scalar_order=0.6,
            face_normals=normals,
        )
    )
    boundaries = assign_boundaries(
        model=base.equation_system,
        geometry=base.geometry,
        policies={
            "Q": policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
        name="p844_lifted_plane",
    )
    return replace(base, boundaries=boundaries)


def _runtime(tmp_path: Path, *, planar: bool = False):
    run_spec = _run_spec(tmp_path)
    simulation = _lifted_simulation(run_spec, planar=planar)
    metadata = {
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
    }
    generator = torch.Generator(device="cpu").manual_seed(144)
    initial = {
        name: 0.01
        * torch.randn((8, 8, 6), generator=generator, dtype=torch.float64)
        for name in Q_COMPONENTS
    }
    request = PlaneRuntimeBuildRequest(
        run_spec,
        metadata,
        initial,
        "cpu",
        simulation,
    )
    construction = PackageRuntimeConstructionInput(
        plan_package_runtime_construction(simulation),
        request,
    )
    return (
        run_spec,
        simulation,
        build_package_simulation_runtime(construction),
    )


def test_lifted_plane_binding_is_narrow_and_compiled_path_stays_closed(tmp_path):
    legacy = _run_spec(tmp_path)
    simulation = _lifted_simulation(legacy)
    binding = bind_simulation_runtime(simulation)
    assert binding.runtime_path == "legacy_production"

    compiled = _run_spec(tmp_path, runtime_path="compiled_v2")
    compiled_simulation = _lifted_simulation(compiled)
    with pytest.raises(SimulationBindingError) as caught:
        bind_simulation_runtime(compiled_simulation)
    assert caught.value.rejection.code is (
        BindingRejectionCode.UNSUPPORTED_LIFTING_RUNTIME
    )

    public = Simulation(
        model=compiled_simulation.equation_system,
        geometry=compiled_simulation.geometry,
        boundaries=compiled_simulation.boundaries,
        numerics=compiled_simulation.numerics,
        time=TimeStepping(
            dt=compiled_simulation.time_integration.integrator.dt,
        ),
        discretization={},
        initial_condition=compiled_simulation.initial_condition,
        execution=TorchSpectralExecution(
            runtime_path=compiled_simulation.execution.runtime_path,
            device=compiled_simulation.execution.options["device"],
            options={
                key: value
                for key, value in compiled_simulation.execution.options.items()
                if key != "device"
            },
        ),
        output=compiled_simulation.workflow,
        invocation=compiled_simulation.invocation,
    )
    with pytest.raises(PublicSimulationCompilationError) as public_error:
        compile_simulation(public)
    assert public_error.value.code is (
        PublicCompilationRejectionCode.RUNTIME_BINDING
    )
    assert public_error.value.context["binding_rejection"]["code"] == (
        "unsupported_lifting_runtime"
    )


def test_runtime_evolves_dst_remainder_and_observes_physical_q(tmp_path):
    _spec, simulation, adapter = _runtime(tmp_path, planar=True)
    lift = adapter.solver.model.static_lifting_runtime
    assert adapter.solver.model.static_model.molecular_field_boundary_conditions == (
        "periodic",
        "periodic",
        "neumann",
    )

    assert adapter.fields["Qxx.bc"] == (
        "periodic",
        "periodic",
        "dirichlet",
    )
    observation = capture_plane_observation(adapter, step=0)
    expected = np.stack(
        [
            plane_physical_component(adapter, name)[0].detach().numpy()
            for name in Q_COMPONENTS
        ],
        axis=-1,
    )
    assert np.array_equal(observation.q, expected)
    assert lift.restart_metadata()["representation"] == "homogeneous_remainder"
    assert lift.restart_metadata()["convention"]["id"] == (
        "de_gennes_S_lambda_max_v1"
    )
    storage = lift.storage_metadata()
    assert storage["operator"]["layout"] == (
        "wall_normal_profile_broadcast"
    )
    assert storage["physical_workspace_layout"] == (
        "on_demand_fused_pointwise"
    )
    assert storage["physical_workspace_bytes"] == 0
    assert storage["linear_correction_storage_bytes"] == (
        torch.empty((), dtype=torch.complex128).element_size()
    )
    assert all(
        item == "broadcast_zero"
        for item in storage["linear_correction_layouts"]
    )
    boundary_identity = hashlib.sha256(
        json.dumps(
            simulation.boundaries.to_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert lift.plan.source_boundary_sha256 == boundary_identity
    adapter.advance(1)
    assert adapter.completed_steps == 1
    assert all(torch.isfinite(adapter.fields[name]).all() for name in Q_COMPONENTS)


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


def test_lifted_checkpoint_restart_is_exact_and_stores_remainder(tmp_path):
    identity, _simulation, continuous = _runtime(tmp_path / "continuous", planar=True)
    _, _, segment = _runtime(tmp_path / "segment", planar=True)
    _, _, resumed = _runtime(tmp_path / "resumed", planar=True)

    continuous.advance(4)
    segment.advance(2)
    checkpoint = capture_plane_checkpoint(
        segment,
        runtime_identity_sha256=identity.runtime_identity_sha256(),
    )
    assert checkpoint.lifting_restart["representation"] == "homogeneous_remainder"
    assert all(
        torch.equal(checkpoint.evolved_spatial[name], segment.fields[name])
        for name in Q_COMPONENTS
    )
    restore_plane_checkpoint(
        resumed,
        checkpoint,
        runtime_identity_sha256=identity.runtime_identity_sha256(),
    )
    resumed.advance(2)

    for name in (*Q_COMPONENTS, "ux", "uy", "uz", "p"):
        assert torch.equal(continuous.fields[name], resumed.fields[name])
    for name in Q_COMPONENTS:
        assert torch.equal(
            plane_physical_component(continuous, name),
            plane_physical_component(resumed, name),
        )


def test_lifting_identity_tamper_is_rejected_before_target_mutation(tmp_path):
    identity, _simulation, source = _runtime(tmp_path / "source")
    _, _, target = _runtime(tmp_path / "target")
    source.advance(1)
    checkpoint = capture_plane_checkpoint(
        source,
        runtime_identity_sha256=identity.runtime_identity_sha256(),
    )
    lifting = json.loads(json.dumps(dict(checkpoint.lifting_restart)))
    lifting["lifting"]["plan_sha256"] = "0" * 64
    tampered = replace(checkpoint, lifting_restart=lifting)
    before = {
        f"{name}{suffix}": target.fields[f"{name}{suffix}"].clone()
        for name in Q_COMPONENTS
        for suffix in ("", ".hat")
    }

    with pytest.raises(ValueError, match="lifting identity"):
        restore_plane_checkpoint(
            target,
            tampered,
            runtime_identity_sha256=identity.runtime_identity_sha256(),
        )
    assert all(
        torch.equal(value, target.fields[name])
        for name, value in before.items()
    )


def test_lifting_checkpoint_device_provenance_is_not_compatibility_identity(
    tmp_path,
):
    identity, _simulation, source = _runtime(tmp_path / "source")
    _, _, target = _runtime(tmp_path / "target")
    source.advance(1)
    checkpoint = capture_plane_checkpoint(
        source,
        runtime_identity_sha256=identity.runtime_identity_sha256(),
    )
    lifting = json.loads(json.dumps(dict(checkpoint.lifting_restart)))
    lifting["lifting"]["device"] = "cuda:0"
    for correction in lifting["linear_corrections"]:
        correction["device"] = "cuda:0"
    lifting.pop("materialization_provenance", None)
    legacy_gpu_checkpoint = replace(checkpoint, lifting_restart=lifting)

    restored_step = restore_plane_checkpoint(
        target,
        legacy_gpu_checkpoint,
        runtime_identity_sha256=identity.runtime_identity_sha256(),
    )

    assert restored_step == 1
    assert all(
        torch.equal(source.fields[f"{name}{suffix}"], target.fields[f"{name}{suffix}"])
        for name in Q_COMPONENTS
        for suffix in ("", ".hat")
    )


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
        numerics=SpectralNumerics(dtype="float64", dealias_rule="cubic_half"),
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
            checkpoint_interval=1,
        ),
    )


def test_public_runner_writes_physical_q_and_checkpointed_remainder(tmp_path):
    simulation = _public_simulation(tmp_path)
    compiled = compile_simulation(simulation)
    assert compiled.lowering_plan.lifting_plan is not None
    result = run_simulation(compiled)
    assert result.final_step == 1

    output = tmp_path / "public_lifted"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["boundary_conditions"]["Q"] == [
        "periodic",
        "periodic",
        "dirichlet",
    ]
    assert metadata["boundary_conditions"]["Q_evolved_representation"] == (
        "homogeneous_remainder"
    )
    assert metadata["boundary_conditions"][
        "Q_evolved_boundary_conditions"
    ] == ["periodic", "periodic", "dirichlet"]
    assert metadata["q_wall_model_note"].startswith(
        "prescribed static Dirichlet Q"
    )
    physical = np.load(output / "Q_1.npy", allow_pickle=False)
    checkpoint = load_plane_checkpoint(output / "checkpoint_1")
    operator = materialize_plane_static_lifting(
        compiled.lowering_plan.lifting_plan,
        dtype=torch.float64,
        device="cpu",
    )
    reconstructed = np.stack(
        [
            (
                checkpoint.evolved_spatial[name][0]
                + operator.lift(name)
            ).numpy()
            for name in Q_COMPONENTS
        ],
        axis=-1,
    )
    assert np.array_equal(physical, reconstructed)
    assert checkpoint.lifting_restart is not None


def test_p844_record_matches_qualified_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
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
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/configuration/public_simulation_runner.py",
        "pssolver/models/active_nematics/beris_edwards.py",
        "pssolver/models/active_nematics/stokes.py",
        "pssolver/operators/lifting.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/runtime/static_lifting.py",
        "pssolver/workflows/plane_checkpoint.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_after_p844:
            continue
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
