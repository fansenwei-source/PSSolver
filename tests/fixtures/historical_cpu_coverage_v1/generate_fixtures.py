"""Generate S-08 fixtures with an exact released PSSolver source tree.

This script is intentionally compatible with the v0.2.0rc1 and v0.2.0rc2
APIs.  Run it with the selected release checkout first on ``PYTHONPATH``.
It never rewrites an existing output directory.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pssolver
import scipy
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
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.functional import (
    build_channel_activity_functional_runtime,
    channel_activity_functional_request,
)
from pssolver.geometries import RectangularChannel
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    Q_COMPONENTS,
    strong_homeotropic_q,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.plane_beris_edwards import PlaneRuntimeBuildRequest
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    write_plane_checkpoint,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tree_sha256(directory: Path) -> str:
    payload = json.dumps(
        _tree(directory), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _torch_build_settings(value: str) -> str:
    settings = [
        line.split("Build settings:", 1)[1].strip()
        for line in value.splitlines()
        if "Build settings:" in line
    ]
    if len(settings) != 1 or not settings[0]:
        raise RuntimeError(
            "torch configuration must contain exactly one Build settings line"
        )
    return settings[0]


def _numpy_build_configuration() -> tuple[str, object]:
    try:
        return "mode_dicts", np.__config__.show(mode="dicts")
    except TypeError:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            np.__config__.show()
        value = stream.getvalue().strip()
        if not value:
            raise RuntimeError("NumPy configuration output is empty")
        return "show_text", value


def numerical_build_fingerprint() -> dict[str, object]:
    """Identify the numerical build without CUDA-visibility probe lines."""

    torch_build_settings = _torch_build_settings(torch.__config__.show())
    numpy_source, numpy_build = _numpy_build_configuration()
    return {
        "fingerprint_schema_version": 2,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_git_version": torch.version.git_version,
        "torch_build_settings_sha256": hashlib.sha256(
            torch_build_settings.encode("utf-8")
        ).hexdigest(),
        "numpy_version": np.__version__,
        "numpy_configuration_source": numpy_source,
        "numpy_configuration_sha256": hashlib.sha256(
            json.dumps(
                numpy_build, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "scipy_version": scipy.__version__,
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


def _channel_fixture(root: Path) -> dict[str, object]:
    shape = (7, 6, 6)
    model = _model()
    geometry = RectangularChannel(shape=shape, lengths=(7.0, 6.0, 6.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    snapshot = root / "snapshot"
    snapshot.mkdir()
    rng = np.random.default_rng(8025)
    values = rng.normal(scale=0.01, size=(*shape, 5)).astype(np.float64)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    np.save(snapshot / "Q_0.npy", values, allow_pickle=False)
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
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
            directory=root / "unused_output",
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
    runtime = build_channel_activity_functional_runtime(
        channel_activity_functional_request(simulation.specification)
    )
    control_spec = runtime.control_specs[0].tensor
    controls = {
        "activity": torch.full(
            control_spec.shape,
            0.01,
            dtype=torch.float64,
            device=control_spec.device,
        )
    }
    state_one = runtime.step(runtime.initial_state(), controls, 0)
    checkpoint = runtime.checkpoint_bridge.export_checkpoint(
        root / "checkpoint", state_one, completed_steps=1
    )
    state_two = runtime.step(state_one, controls, 1)
    oracle = root / "continuation_oracle"
    oracle.mkdir()
    np.save(
        oracle / "q_physical_step_2.npy",
        state_two[0].detach().cpu().contiguous().numpy(),
        allow_pickle=False,
    )
    np.save(
        oracle / "q_spectral_step_2.npy",
        state_two[1].detach().cpu().contiguous().numpy(),
        allow_pickle=False,
    )
    return {
        "case": "channel_functional_odd_periodic_axis",
        "shape": list(shape),
        "periodic_axis": 0,
        "periodic_axis_size": shape[0],
        "completed_steps": 1,
        "continuation_oracle_step": 2,
        "snapshot_sha256": _sha256(snapshot / "Q_0.npy"),
        "checkpoint_tree_sha256": _tree_sha256(checkpoint),
        "continuation_tree_sha256": _tree_sha256(oracle),
    }


def _plane_run_spec(root: Path):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=root / "unused_output",
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


def _plane_fixture(root: Path) -> dict[str, object]:
    run_spec = _plane_run_spec(root)
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    normals = {
        (2, "lower"): (0.0, 0.0, -1.0),
        (2, "upper"): (0.0, 0.0, 1.0),
    }
    simulation = replace(
        base,
        boundaries=assign_boundaries(
            model=base.equation_system,
            geometry=base.geometry,
            policies={
                "Q": strong_homeotropic_q(
                    scalar_order=0.6,
                    face_normals=normals,
                ),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
            name="s08_authentic_rc1_lifted_plane",
        ),
    )
    coordinate = torch.arange(7 * 7 * 6, dtype=torch.float64).reshape(7, 7, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    initial = {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }
    request = PlaneRuntimeBuildRequest(
        run_spec,
        {
            "configuration": run_spec.identity_metadata(),
            "runtime_selection": run_spec.runtime_selection_metadata(),
        },
        initial,
        "cpu",
        simulation,
    )
    adapter = build_package_simulation_runtime(
        PackageRuntimeConstructionInput(
            plan_package_runtime_construction(simulation),
            request,
        )
    )
    adapter.advance(1)
    checkpoint = capture_plane_checkpoint(
        adapter,
        runtime_identity_sha256=run_spec.runtime_identity_sha256(),
    )
    assert checkpoint.lifting_restart is not None
    target = write_plane_checkpoint(root / "checkpoint", checkpoint)
    return {
        "case": "plane_rc1_static_lifting",
        "shape": [7, 7, 6],
        "completed_steps": 1,
        "runtime_identity_sha256": run_spec.runtime_identity_sha256(),
        "lifting_restart_present": True,
        "checkpoint_tree_sha256": _tree_sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("channel_odd", "plane_lifted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = (
        _channel_fixture(output)
        if args.case == "channel_odd"
        else _plane_fixture(output)
    )
    payload["source"] = {
        "commit": args.source_commit,
        "pssolver_version": pssolver.__version__,
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "numerical_build_fingerprint": numerical_build_fingerprint(),
    }
    _write_json(output / "generation.json", payload)


if __name__ == "__main__":
    main()
