"""Installed-wheel closure gate for the S-07 package surface.

This file is intentionally outside ``tests/``.  Qualification copies it next
to an isolated D-0008 environment and runs it explicitly against the installed
wheel with no repository root on ``sys.path``.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.metadata
import json
from pathlib import Path
import sysconfig

import numpy as np
import pytest
import torch

import pssolver
from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PeriodicBox, PlaneSlab, RectangularChannel
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    create_initial_condition,
)
from pssolver.runtime.channel_active_nematics import ChannelRuntimeBuildRequest
from pssolver.runtime.channel_application_bridge import (
    build_package_compiled_channel_runtime,
    build_package_legacy_channel_runtime,
)


PUBLIC_COMPILATION_KEYS = {
    "schema_version",
    "compilation_identity_sha256",
    "source_simulation_sha256",
    "application",
    "application_simulation_sha256",
    "application_request_sha256",
    "normalization",
    "fallback_allowed",
}


def _model() -> CompleteStressBerisEdwards:
    return CompleteStressBerisEdwards(
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


def _snapshot(tmp_path: Path, seed: int) -> SnapshotInitialCondition:
    directory = tmp_path / f"snapshot_{seed}"
    directory.mkdir()
    values = np.zeros((8, 8, 8, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    values += np.random.default_rng(seed).normal(scale=1.0e-3, size=values.shape)
    np.save(directory / "Q_0.npy", values, allow_pickle=False)
    return SnapshotInitialCondition(directory, step=0)


def _plane(tmp_path: Path) -> Simulation:
    model = _model()
    geometry = PlaneSlab(shape=(8, 8, 8), lengths=(8.0, 8.0, 8.0))
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
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
                "defect_min_separation": 1.0,
                "defect_core_radius": 0.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
                "initial_s": 1.0 / 3.0,
            },
        ),
        execution=TorchSpectralExecution(
            runtime_path="compiled_v2",
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
            directory=tmp_path / "plane",
            steps=1,
            save_start_step=0,
            save_interval=1,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        invocation=InvocationSpec({"dry_run": True}),
    )


def _complete_stress(tmp_path: Path, *, periodic: bool) -> Simulation:
    model = _model()
    geometry = (
        PeriodicBox(shape=(8, 8, 8), lengths=(8.0, 8.0, 8.0))
        if periodic
        else RectangularChannel(shape=(8, 8, 8), lengths=(8.0, 8.0, 8.0))
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity() if periodic else no_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            projected_transform_execution="truncated",
            spectral_storage="hermitian_half" if periodic else "full_complex",
            hermitian_axis=1 if periodic else None,
        ),
        time=TimeStepping(dt=0.001),
        initial_condition=_snapshot(tmp_path, 31 if periodic else 37),
        execution=TorchSpectralExecution(
            runtime_path=(
                "periodic_spectral" if periodic else "channel_complete_stress"
            ),
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral" if periodic else "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / ("periodic" if periodic else "channel_complete"),
            steps=1,
            save_start_step=0,
            save_interval=1,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=(
            {}
            if periodic
            else {
                "pressure_solver": {
                    "algorithm": "preconditioned_conjugate_gradient",
                    "relative_tolerance": 1.0e-10,
                    "max_iterations": 40,
                    "fixed_iterations": 12,
                    "warm_start": True,
                }
            }
        ),
        invocation=InvocationSpec({"dry_run": True}),
    )


def _legacy(tmp_path: Path) -> Simulation:
    request = ChannelActiveNematicRunSpec(
        shape=(8, 6, 6),
        lengths=(8.0, 6.0, 6.0),
        steps=1,
        save_interval=1,
        diagnostic_interval=1,
        generated_output_directory=tmp_path / "legacy",
        snapshot_output_directory=tmp_path / "legacy_snapshot",
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=40,
        pressure_fixed_iterations=8,
        device="cpu",
        runtime_path="legacy_channel",
    )
    source = compose_channel_active_nematics_simulation(request.components)
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        discretization=source.discretization_parameters,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        invocation=InvocationSpec({"dry_run": True}),
    )


def _module_names() -> list[str]:
    names: set[str] = set()
    for entry in importlib.metadata.files("pssolver") or ():
        path = Path(str(entry))
        if path.suffix != ".py" or ".dist-info" in str(path):
            continue
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        names.add(".".join(parts))
    return sorted(names)


def test_every_shipped_python_module_imports() -> None:
    assert Path(pssolver.__file__).resolve().is_relative_to(
        Path(sysconfig.get_paths()["purelib"]).resolve()
    )
    allowed_optional_failure = []
    for name in _module_names():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if name == "pssolver.visualizer" and exc.name == "pygame":
                allowed_optional_failure.append(name)
                continue
            raise
    assert allowed_optional_failure in ([], ["pssolver.visualizer"])


@pytest.mark.parametrize(
    ("runtime_path", "builder"),
    (
        ("legacy_channel", build_package_legacy_channel_runtime),
        ("compiled_channel_v2", build_package_compiled_channel_runtime),
    ),
)
def test_installed_wheel_builds_both_channel_runtimes(runtime_path, builder) -> None:
    spec = ChannelActiveNematicRunSpec(
        shape=(8, 6, 6),
        lengths=(8.0, 6.0, 6.0),
        steps=1,
        save_interval=1,
        diagnostic_interval=1,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=40,
        pressure_fixed_iterations=8,
        device="cpu",
        runtime_path=runtime_path,
    )
    initial_q = create_initial_condition(
        "aligned_x_smooth_noise",
        shape=spec.shape,
        boundary_conditions=spec.boundaries.q,
        seed=spec.seed,
        S_initial=spec.initial_s,
        noise_theta=spec.noise_theta,
        noise_phi=spec.noise_phi,
        sigma_x=spec.smoothing_sigma[0],
        sigma_y=spec.smoothing_sigma[1],
        sigma_z=spec.smoothing_sigma[2],
        dtype=torch.float32,
    )
    request = ChannelRuntimeBuildRequest(
        spec,
        {
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_q,
        torch.device("cpu"),
    )
    assert builder(request) is not None


@pytest.mark.parametrize("factory", (_plane, _legacy))
def test_installed_wheel_public_dry_runs_emit_complete_compilation(
    factory,
    tmp_path,
    capsys,
) -> None:
    source = factory(tmp_path)
    compiled = compile_simulation(source)
    result = run_simulation(compiled, emit_metadata=True)
    assert result.status.value == "dry_run"
    public = json.loads(capsys.readouterr().out)["public_compilation"]
    assert set(public) == PUBLIC_COMPILATION_KEYS
    assert public["compilation_identity_sha256"] == compiled.compilation_identity_sha256


@pytest.mark.parametrize("periodic", (True, False))
def test_installed_wheel_complete_stress_dry_runs_emit_complete_compilation(
    periodic,
    tmp_path,
    capsys,
) -> None:
    source = _complete_stress(tmp_path, periodic=periodic)
    compiled = compile_simulation(source)
    result = run_simulation(compiled, emit_metadata=True)
    assert result.status.value == "dry_run"
    public = json.loads(capsys.readouterr().out)["public_compilation"]
    assert set(public) == PUBLIC_COMPILATION_KEYS
    assert public["compilation_identity_sha256"] == compiled.compilation_identity_sha256
