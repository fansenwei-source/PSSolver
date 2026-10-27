"""Generate and replay the immutable v0.2.0 checkpoint compatibility corpus.

The generator is intentionally executable from an installed PSSolver wheel:
run this file by absolute path while the repository root is absent from
``PYTHONPATH``.  The committed manifest records the imported package path and
the published wheel digest.  Tests import the same construction helpers from
the source tree and never modify the frozen checkpoint directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import torch

import pssolver
from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.applications.channel_beris_edwards import load_channel_initial_q
from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.channel import build_active_nematic_channel
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.functional import (
    build_channel_activity_functional_runtime,
    build_functional_runtime,
    channel_activity_functional_request,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    Q_COMPONENTS,
    VELOCITY_COMPONENTS,
)
from pssolver.runtime import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)
from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows.channel_beris_edwards import (
    _load_checkpoint as load_channel_complete_stress_checkpoint,
    write_channel_beris_edwards_checkpoint,
)
from pssolver.workflows.channel_checkpoint import (
    capture_channel_checkpoint,
    load_channel_checkpoint,
    restore_channel_checkpoint,
    write_channel_checkpoint,
)
from pssolver.workflows.periodic_checkpoint import (
    capture_periodic_checkpoint,
    load_periodic_checkpoint,
    restore_periodic_checkpoint,
    write_periodic_checkpoint,
)
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    load_plane_checkpoint,
    restore_plane_checkpoint,
    write_plane_checkpoint,
)


FIXTURE_SCHEMA_VERSION = 1
RELEASE_VERSION = "0.2.0"
RELEASE_TAG = "v0.2.0"
RELEASE_COMMIT = "6d2934e73b543f6652d112733657775333e0f396"
PUBLISHED_WHEEL_SHA256 = (
    "ab4beef1a347568e8ad4162291feb62e94e179849853ca1417583634649a1151"
)
CONTINUATION_ORACLE_SCHEMA_VERSION = 1
CONTINUATION_TOLERANCE_EPS_MULTIPLIER = 8.0
NUMERICAL_BUILD_FINGERPRINT_SCHEMA_VERSION = 2
PERIODIC_SHAPE = (6, 6, 4)
CHANNEL_SHAPE = (7, 6, 6)
PLANE_SHAPE = (7, 7, 6)
LEGACY_CHANNEL_SHAPE = (7, 6, 5)


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
        _tree(directory),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_record(value: torch.Tensor) -> dict[str, object]:
    array = _tensor_array(value)
    return _array_record(array)


def _tensor_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu").contiguous().numpy().copy()


def _array_record(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


_CUDA_VISIBILITY_ONLY_TORCH_CONFIG_PREFIXES = (
    "- CUDA Runtime",
    "- NVCC architecture flags",
    "- CuDNN",
    "- Magma",
)


def _visibility_invariant_torch_configuration(value: str) -> str:
    """Remove probe-only lines whose presence depends on CUDA visibility.

    The retained build-settings line still records whether Torch was built
    with CUDA, its CUDA/CuDNN versions, compiler, BLAS, MKL and CPU features.
    Only the four human-readable probe summaries disappear when no GPU is
    visible, so they cannot decide whether a CPU continuation is bitwise.
    """

    lines = [
        line
        for line in value.splitlines()
        if not line.strip().startswith(
            _CUDA_VISIBILITY_ONLY_TORCH_CONFIG_PREFIXES
        )
    ]
    normalized = "\n".join(lines)
    if value.endswith("\n"):
        normalized += "\n"
    return normalized


def numerical_build_fingerprint() -> dict[str, object]:
    """Identify the numerical build that controls exact FFT continuation."""

    torch_configuration = _visibility_invariant_torch_configuration(
        torch.__config__.show()
    )
    numpy_build = np.__config__.show(mode="dicts")
    return {
        "fingerprint_schema_version": (
            NUMERICAL_BUILD_FINGERPRINT_SCHEMA_VERSION
        ),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_configuration_sha256": hashlib.sha256(
            torch_configuration.encode("utf-8")
        ).hexdigest(),
        "numpy_version": np.__version__,
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


def _snapshot(root: Path, name: str, shape: tuple[int, int, int]) -> Path:
    directory = root / "construction" / name
    directory.mkdir(parents=True, exist_ok=True)
    values = np.zeros((*shape, len(Q_COMPONENTS)), dtype=np.float64)
    coordinate = np.arange(values.size, dtype=np.float64).reshape(values.shape)
    coordinate -= coordinate.mean()
    coordinate /= max(1, coordinate.size)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    values += coordinate * 1.0e-4
    np.save(directory / "Q_0.npy", values, allow_pickle=False)
    return directory


def _output(root: Path, name: str) -> Output:
    return Output(
        directory=root / "outputs" / name,
        steps=2,
        save_interval=2,
        diagnostic_interval=2,
        save_start_step=0,
        diagnostics=True,
        save_hydrodynamics=True,
    )


def periodic_simulation(root: Path) -> Simulation:
    model = _model()
    geometry = PeriodicBox(
        shape=PERIODIC_SHAPE,
        lengths=tuple(float(value) for value in PERIODIC_SHAPE),
    )
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
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            _snapshot(root, "periodic", PERIODIC_SHAPE),
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=_output(root, "periodic"),
    )


def channel_simulation(root: Path) -> Simulation:
    model = _model()
    geometry = RectangularChannel(
        shape=CHANNEL_SHAPE,
        lengths=tuple(float(value) for value in CHANNEL_SHAPE),
    )
    return Simulation(
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
        time=TimeStepping(dt=0.001),
        initial_condition=SnapshotInitialCondition(
            _snapshot(root, "channel_complete_stress", CHANNEL_SHAPE),
            step=0,
        ),
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
        output=_output(root, "channel_complete_stress"),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
        invocation=InvocationSpec({}),
    )


def build_periodic_functional(root: Path):
    request = periodic_activity_functional_request(
        periodic_simulation(root).specification
    )
    return build_functional_runtime(request)


def build_channel_functional(root: Path):
    request = channel_activity_functional_request(
        channel_simulation(root).specification
    )
    return build_channel_activity_functional_runtime(request)


def _control(runtime, value: float = 0.01) -> dict[str, torch.Tensor]:
    spec = runtime.control_specs[0].tensor
    return {
        "activity": torch.full(
            spec.shape,
            value,
            dtype=runtime.initial_state()[0].dtype,
            device=spec.device,
        )
    }


def build_periodic_production(root: Path):
    compiled = compile_simulation(periodic_simulation(root))
    run_spec = compiled.application_request
    initial, _, _ = load_periodic_initial_q(run_spec)
    adapter = build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(
            run_spec=run_spec,
            initial_values=initial,
            device="cpu",
        )
    )
    return run_spec, adapter


def build_channel_complete_stress(root: Path):
    compiled = compile_simulation(channel_simulation(root))
    run_spec = compiled.application_request
    initial, _, _ = load_channel_initial_q(run_spec)
    adapter = build_channel_beris_edwards_runtime(
        ChannelBerisEdwardsRuntimeBuildRequest(
            run_spec=run_spec,
            initial_values=initial,
            device="cpu",
        )
    )
    return run_spec, adapter


def build_plane_legacy(root: Path):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=root / "outputs" / "plane_legacy",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        runtime_path="legacy_production",
        nx=PLANE_SHAPE[0],
        ny=PLANE_SHAPE[1],
        nz=PLANE_SHAPE[2],
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
    coordinate = torch.arange(
        np.prod(PLANE_SHAPE), dtype=torch.float64
    ).reshape(PLANE_SHAPE)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    initial = {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }
    solver, projector = build_legacy_plane_runtime(
        run_spec,
        device="cpu",
        initial_values=initial,
    )
    return run_spec, LegacyPlaneRuntimeAdapter(solver, projector)


def _legacy_channel_initial() -> dict[str, torch.Tensor]:
    coordinate = torch.arange(
        np.prod(LEGACY_CHANNEL_SHAPE), dtype=torch.float32
    ).reshape(LEGACY_CHANNEL_SHAPE)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def build_legacy_channel(_root: Path):
    run_spec = ChannelActiveNematicRunSpec(
        shape=LEGACY_CHANNEL_SHAPE,
        lengths=tuple(float(value) for value in LEGACY_CHANNEL_SHAPE),
        dt=1.0e-3,
        steps=2,
        save_interval=2,
        diagnostic_interval=2,
        device="cpu",
        dtype="float32",
        activity=0.2,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=100,
        pressure_fixed_iterations=8,
        runtime_path="legacy_channel",
        seed=24,
    )
    values = _legacy_channel_initial()
    metadata = {
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
    }
    request = ChannelRuntimeBuildRequest(run_spec, metadata, values, "cpu")

    def builder():
        material = run_spec.components.material
        pressure = run_spec.components.pressure_solver
        solver = build_active_nematic_channel(
            run_spec.shape,
            run_spec.lengths,
            run_spec.dt,
            values,
            device="cpu",
            batchsize=run_spec.batch_size,
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
            (run_spec.batch_size, *run_spec.shape),
            material.activity,
            dtype=solver.dtype,
            device=solver.device,
        )
        return solver

    adapter = build_channel_active_nematic_runtime(
        request,
        legacy_builder=builder,
    )
    return run_spec, adapter


def _field_records(adapter, components: tuple[str, ...]) -> dict[str, object]:
    records = {}
    for name in components:
        records[name] = _tensor_record(adapter.fields[name])
        records[f"{name}.hat"] = _tensor_record(adapter.fields[f"{name}.hat"])
    records["completed_steps"] = adapter.completed_steps
    records["integrator_step_count"] = int(adapter.solver.integrator.step_count)
    records["integrator_refresh_count"] = int(
        adapter.solver.integrator.refresh_count
    )
    return records


def _field_arrays(
    adapter, components: tuple[str, ...]
) -> dict[str, np.ndarray]:
    arrays = {}
    for name in components:
        arrays[name] = _tensor_array(adapter.fields[name])
        arrays[f"{name}.hat"] = _tensor_array(adapter.fields[f"{name}.hat"])
    return arrays


def functional_continuation_records(state) -> dict[str, object]:
    return {
        "q_physical": _tensor_record(state[0]),
        "q_spectral": _tensor_record(state[1]),
        "completed_steps": 2,
    }


def functional_continuation_arrays(state) -> dict[str, np.ndarray]:
    return {
        "q_physical": _tensor_array(state[0]),
        "q_spectral": _tensor_array(state[1]),
    }


def periodic_continuation_records(adapter) -> dict[str, object]:
    return _field_records(adapter, tuple(Q_COMPONENTS))


def periodic_continuation_arrays(adapter) -> dict[str, np.ndarray]:
    return _field_arrays(adapter, tuple(Q_COMPONENTS))


def channel_complete_continuation_records(adapter) -> dict[str, object]:
    records = _field_records(
        adapter,
        (*tuple(Q_COMPONENTS), *tuple(VELOCITY_COMPONENTS), "p"),
    )
    records["pressure_guess"] = _tensor_record(adapter.capture_pressure_guess())
    return records


def channel_complete_continuation_arrays(adapter) -> dict[str, np.ndarray]:
    arrays = _field_arrays(
        adapter,
        (*tuple(Q_COMPONENTS), *tuple(VELOCITY_COMPONENTS), "p"),
    )
    arrays["pressure_guess"] = _tensor_array(adapter.capture_pressure_guess())
    return arrays


def plane_continuation_records(adapter) -> dict[str, object]:
    return _field_records(adapter, tuple(Q_COMPONENTS))


def plane_continuation_arrays(adapter) -> dict[str, np.ndarray]:
    return _field_arrays(adapter, tuple(Q_COMPONENTS))


def legacy_channel_continuation_records(adapter) -> dict[str, object]:
    records = _field_records(
        adapter,
        (*tuple(Q_COMPONENTS), *tuple(VELOCITY_COMPONENTS), "p"),
    )
    records["pressure_guess"] = _tensor_record(adapter.capture_pressure_guess())
    return records


def legacy_channel_continuation_arrays(adapter) -> dict[str, np.ndarray]:
    arrays = _field_arrays(
        adapter,
        (*tuple(Q_COMPONENTS), *tuple(VELOCITY_COMPONENTS), "p"),
    )
    arrays["pressure_guess"] = _tensor_array(adapter.capture_pressure_guess())
    return arrays


def _case_record(directory: Path, continuation: dict[str, object]) -> dict[str, object]:
    files = _tree(directory)
    return {
        "checkpoint_directory": directory.name,
        "completed_steps": 1,
        "continuation_steps": 1,
        "file_count": len(files),
        "tree_sha256": _tree_sha256(directory),
        "files": files,
        "continuation": continuation,
    }


def generate(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"fixture output already exists: {output}")
    if pssolver.__version__ != RELEASE_VERSION:
        raise RuntimeError(
            f"fixture generator requires PSSolver {RELEASE_VERSION}, got "
            f"{pssolver.__version__}"
        )
    output.mkdir(parents=True)
    work = output / ".generation_work"
    work.mkdir()
    cases: dict[str, dict[str, object]] = {}

    runtime = build_periodic_functional(work / "functional_periodic")
    controls = _control(runtime)
    state = runtime.step(runtime.initial_state(), controls, 0)
    directory = runtime.checkpoint_bridge.export_checkpoint(
        output / "functional_periodic", state, completed_steps=1
    )
    continued = runtime.step(state, controls, 1)
    cases["functional_periodic"] = _case_record(
        directory, functional_continuation_records(continued)
    )

    runtime = build_channel_functional(work / "functional_channel")
    controls = _control(runtime)
    state = runtime.step(runtime.initial_state(), controls, 0)
    directory = runtime.checkpoint_bridge.export_checkpoint(
        output / "functional_channel", state, completed_steps=1
    )
    continued = runtime.step(state, controls, 1)
    cases["functional_channel"] = _case_record(
        directory, functional_continuation_records(continued)
    )

    run_spec, adapter = build_periodic_production(work / "production_periodic")
    adapter.advance(1)
    directory = write_periodic_checkpoint(
        output / "production_periodic",
        capture_periodic_checkpoint(adapter, run_spec=run_spec),
    )
    adapter.advance(1)
    cases["production_periodic"] = _case_record(
        directory, periodic_continuation_records(adapter)
    )

    run_spec, adapter = build_channel_complete_stress(
        work / "production_channel_complete_stress"
    )
    adapter.advance(1)
    directory = output / "production_channel_complete_stress"
    write_channel_beris_edwards_checkpoint(
        directory,
        adapter,
        run_spec=run_spec,
    )
    adapter.advance(1)
    cases["production_channel_complete_stress"] = _case_record(
        directory, channel_complete_continuation_records(adapter)
    )

    run_spec, adapter = build_plane_legacy(work / "production_plane_legacy")
    adapter.advance(1)
    directory = write_plane_checkpoint(
        output / "production_plane_legacy",
        capture_plane_checkpoint(adapter, run_spec=run_spec),
    )
    adapter.advance(1)
    cases["production_plane_legacy"] = _case_record(
        directory, plane_continuation_records(adapter)
    )

    run_spec, adapter = build_legacy_channel(
        work / "production_legacy_channel"
    )
    adapter.advance(1)
    directory = write_channel_checkpoint(
        output / "production_legacy_channel",
        capture_channel_checkpoint(
            adapter,
            runtime_identity_sha256=run_spec.runtime_identity_sha256(),
        ),
    )
    adapter.advance(1)
    cases["production_legacy_channel"] = _case_record(
        directory, legacy_channel_continuation_records(adapter)
    )

    # Construction-only scratch is not part of the frozen corpus.
    import shutil

    shutil.rmtree(work)
    manifest = {
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "release": {
            "version": RELEASE_VERSION,
            "tag": RELEASE_TAG,
            "commit": RELEASE_COMMIT,
            "published_wheel_sha256": PUBLISHED_WHEEL_SHA256,
            "pssolver_import": str(Path(pssolver.__file__).resolve()),
            "source_shadow_import": False,
        },
        "generation_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "policy": {
            "read_only_fixture": True,
            "direct_import_required": True,
            "continuation_equivalence": "bitwise",
            "continuation_steps": 1,
            "fixture_mutation_allowed": False,
        },
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _assert_records(actual: dict[str, object], expected: dict[str, object]) -> None:
    if actual != expected:
        raise AssertionError("continued checkpoint state differs from v0.2.0")


def _assert_exact_payload(
    directory: Path,
    restored: dict[str, torch.Tensor | np.ndarray],
) -> dict[str, object]:
    payloads = sorted(directory.glob("*.npy"))
    expected_names = {path.name for path in payloads}
    if set(restored) != expected_names:
        raise AssertionError(
            f"restored payload names differ: {set(restored)!r} != "
            f"{expected_names!r}"
        )
    records = {}
    for path in payloads:
        expected = np.load(path, allow_pickle=False)
        value = restored[path.name]
        actual = _tensor_array(value) if isinstance(value, torch.Tensor) else value
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"restored payload layout differs for {path.name}")
        if not np.array_equal(actual, expected):
            raise AssertionError(f"restored payload bytes differ for {path.name}")
        records[path.name] = _array_record(actual)
    return {
        "equivalence": "bitwise",
        "payload_count": len(records),
        "payloads": records,
    }


def _component_payloads(
    *,
    spatial: dict[str, torch.Tensor],
    spectral: dict[str, torch.Tensor],
    spatial_prefix: str,
    spectral_prefix: str,
    extra: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    payloads = {
        f"{spatial_prefix}__{name}.npy": value
        for name, value in spatial.items()
    }
    payloads.update(
        {
            f"{spectral_prefix}__{name}.npy": value
            for name, value in spectral.items()
        }
    )
    if extra:
        payloads.update(extra)
    return payloads


def _run_replay_case(
    case: str, fixture_root: Path, work_root: Path
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    directory = fixture_root / case
    if case == "functional_periodic":
        runtime = build_periodic_functional(work_root)
        restored = runtime.checkpoint_bridge.import_checkpoint(directory)
        payloads = {}
        for index, name in enumerate(Q_COMPONENTS):
            payloads[f"spatial__{name}.npy"] = restored.state[0][index]
            payloads[f"spectral__{name}.npy"] = restored.state[1][index]
        payload_report = _assert_exact_payload(directory, payloads)
        continued = runtime.step(restored.state, _control(runtime), 1)
        actual = functional_continuation_records(continued)
        arrays = functional_continuation_arrays(continued)
    elif case == "functional_channel":
        runtime = build_channel_functional(work_root)
        restored = runtime.checkpoint_bridge.import_checkpoint(directory)
        payload_report = _assert_exact_payload(
            directory,
            {
                "state__q_physical.npy": restored.state[0],
                "state__q_spectral.npy": restored.state[1],
            },
        )
        continued = runtime.step(restored.state, _control(runtime), 1)
        actual = functional_continuation_records(continued)
        arrays = functional_continuation_arrays(continued)
    elif case == "production_periodic":
        run_spec, adapter = build_periodic_production(work_root)
        checkpoint = load_periodic_checkpoint(directory)
        payload_report = _assert_exact_payload(
            directory,
            _component_payloads(
                spatial=dict(checkpoint.evolved_spatial),
                spectral=dict(checkpoint.evolved_spectral),
                spatial_prefix="spatial",
                spectral_prefix="spectral",
            ),
        )
        restore_periodic_checkpoint(adapter, checkpoint, run_spec=run_spec)
        adapter.advance(1)
        actual = periodic_continuation_records(adapter)
        arrays = periodic_continuation_arrays(adapter)
    elif case == "production_channel_complete_stress":
        run_spec, adapter = build_channel_complete_stress(work_root)
        load_channel_complete_stress_checkpoint(
            directory, adapter, run_spec=run_spec
        )
        fields = adapter.fields
        components = (*tuple(Q_COMPONENTS), *tuple(VELOCITY_COMPONENTS), "p")
        payload_report = _assert_exact_payload(
            directory,
            _component_payloads(
                spatial={name: fields[name] for name in components},
                spectral={name: fields[f"{name}.hat"] for name in components},
                spatial_prefix="spatial",
                spectral_prefix="spectral",
                extra={
                    "backend__pressure_guess.npy": adapter.capture_pressure_guess()
                },
            ),
        )
        adapter.advance(1)
        actual = channel_complete_continuation_records(adapter)
        arrays = channel_complete_continuation_arrays(adapter)
    elif case == "production_plane_legacy":
        run_spec, adapter = build_plane_legacy(work_root)
        checkpoint = load_plane_checkpoint(directory)
        payload_report = _assert_exact_payload(
            directory,
            _component_payloads(
                spatial=dict(checkpoint.evolved_spatial),
                spectral=dict(checkpoint.evolved_spectral),
                spatial_prefix="evolved_spatial",
                spectral_prefix="evolved_spectral",
            ),
        )
        restore_plane_checkpoint(adapter, checkpoint, run_spec=run_spec)
        adapter.advance(1)
        actual = plane_continuation_records(adapter)
        arrays = plane_continuation_arrays(adapter)
    elif case == "production_legacy_channel":
        run_spec, adapter = build_legacy_channel(work_root)
        checkpoint = load_channel_checkpoint(directory)
        payload_report = _assert_exact_payload(
            directory,
            _component_payloads(
                spatial=dict(checkpoint.evolved_spatial),
                spectral=dict(checkpoint.evolved_spectral),
                spatial_prefix="evolved_spatial",
                spectral_prefix="evolved_spectral",
                extra={
                    **{
                        f"algebraic_spatial__{name}.npy": value
                        for name, value in checkpoint.algebraic_spatial.items()
                    },
                    **{
                        f"algebraic_spectral__{name}.npy": value
                        for name, value in checkpoint.algebraic_spectral.items()
                    },
                    "persistent__pressure_guess.npy": checkpoint.pressure_guess,
                },
            ),
        )
        restore_channel_checkpoint(
            adapter,
            checkpoint,
            runtime_identity_sha256=run_spec.runtime_identity_sha256(),
        )
        adapter.advance(1)
        actual = legacy_channel_continuation_records(adapter)
        arrays = legacy_channel_continuation_arrays(adapter)
    else:
        raise ValueError(f"unknown fixture case: {case}")
    return actual, arrays, payload_report


def _oracle_array_filename(name: str) -> str:
    return name.replace(".", "__") + ".npy"


def generate_continuation_oracle(
    fixture_root: Path, output: Path
) -> dict[str, object]:
    """Write reference continuations separately from the immutable checkpoints."""

    if output.exists():
        raise FileExistsError(f"continuation oracle already exists: {output}")
    fixture_manifest = json.loads(
        (fixture_root / "manifest.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True)
    work = output / ".generation_work"
    work.mkdir()
    cases = {}
    for case in sorted(fixture_manifest["cases"]):
        actual, arrays, payload_report = _run_replay_case(
            case, fixture_root, work / case
        )
        expected = fixture_manifest["cases"][case]["continuation"]
        _assert_records(actual, expected)
        case_directory = output / case
        case_directory.mkdir()
        array_records = {}
        for name, array in sorted(arrays.items()):
            filename = _oracle_array_filename(name)
            path = case_directory / filename
            np.save(path, array, allow_pickle=False)
            array_records[name] = {
                **_array_record(array),
                "file": f"{case}/{filename}",
                "file_sha256": _sha256(path),
            }
        cases[case] = {
            "arrays": array_records,
            "payload_validation": payload_report,
        }
    import shutil

    shutil.rmtree(work)
    manifest = {
        "oracle_schema_version": CONTINUATION_ORACLE_SCHEMA_VERSION,
        "checkpoint_fixture_tree_sha256": _tree_sha256(fixture_root),
        "numerical_build_fingerprint": numerical_build_fingerprint(),
        "tolerance_policy": {
            "matching_build": "bitwise",
            "other_builds": "dtype_epsilon",
            "epsilon_multiplier": CONTINUATION_TOLERANCE_EPS_MULTIPLIER,
        },
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _real_dtype(dtype: np.dtype) -> np.dtype:
    return np.empty((), dtype=dtype).real.dtype


def _compare_continuation(
    actual_records: dict[str, object],
    actual_arrays: dict[str, np.ndarray],
    expected_records: dict[str, object],
    oracle_root: Path,
    case: str,
) -> dict[str, object]:
    oracle_manifest = json.loads(
        (oracle_root / "manifest.json").read_text(encoding="utf-8")
    )
    reference_fingerprint = oracle_manifest["numerical_build_fingerprint"]
    observed_fingerprint = numerical_build_fingerprint()
    matching_build = observed_fingerprint == reference_fingerprint
    mode = "bitwise" if matching_build else "dtype_epsilon"
    oracle_case = oracle_manifest["cases"][case]["arrays"]
    if set(actual_arrays) != set(oracle_case):
        raise AssertionError("continuation array names differ from the oracle")

    comparisons = {}
    for name, actual in sorted(actual_arrays.items()):
        record = oracle_case[name]
        path = oracle_root / record["file"]
        if _sha256(path) != record["file_sha256"]:
            raise AssertionError(f"continuation oracle checksum differs for {name}")
        expected = np.load(path, allow_pickle=False)
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"continuation layout differs for {name}")
        if not np.isfinite(actual).all():
            raise AssertionError(f"continuation is non-finite for {name}")
        epsilon = float(np.finfo(_real_dtype(actual.dtype)).eps)
        tolerance = CONTINUATION_TOLERANCE_EPS_MULTIPLIER * epsilon
        difference = np.abs(actual - expected)
        max_abs = float(difference.max(initial=0.0))
        if matching_build:
            passed = np.array_equal(actual, expected)
        else:
            passed = np.allclose(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
                equal_nan=False,
            )
        if not passed:
            raise AssertionError(
                f"continued checkpoint state differs for {name}: "
                f"mode={mode}, max_abs={max_abs}, tolerance={tolerance}"
            )
        expected_layout = expected_records[name]
        if expected_layout["shape"] != list(actual.shape) or (
            expected_layout["dtype"] != str(actual.dtype)
        ):
            raise AssertionError(f"continuation manifest layout differs for {name}")
        if matching_build and actual_records[name] != expected_layout:
            raise AssertionError(
                f"matching-build continuation digest differs for {name}"
            )
        comparisons[name] = {
            "dtype": str(actual.dtype),
            "epsilon": epsilon,
            "rtol": 0.0 if matching_build else tolerance,
            "atol": 0.0 if matching_build else tolerance,
            "max_abs_error": max_abs,
            "passed": True,
        }

    scalar_keys = set(expected_records) - set(actual_arrays)
    if set(actual_records) - set(actual_arrays) != scalar_keys:
        raise AssertionError("continuation scalar record names differ")
    for name in scalar_keys:
        if actual_records[name] != expected_records[name]:
            raise AssertionError(f"continuation scalar differs for {name}")
    return {
        "mode": mode,
        "matching_numerical_build": matching_build,
        "reference_build": reference_fingerprint,
        "observed_build": observed_fingerprint,
        "arrays": comparisons,
    }


def replay_case(
    case: str,
    fixture_root: Path,
    work_root: Path,
    oracle_root: Path | None = None,
) -> dict[str, object]:
    manifest = json.loads(
        (fixture_root / "manifest.json").read_text(encoding="utf-8")
    )
    expected = manifest["cases"][case]["continuation"]
    actual, arrays, payload_report = _run_replay_case(
        case, fixture_root, work_root
    )
    if oracle_root is None:
        oracle_root = (
            fixture_root.parent.parent
            / "oracles"
            / "v0_2_0_checkpoint_continuations_v1"
        )
    continuation_report = _compare_continuation(
        actual, arrays, expected, oracle_root, case
    )
    return {
        "case": case,
        "payload": payload_report,
        "continuation": continuation_report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--continuation-oracle", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.continuation_oracle:
        if args.fixture_root is None:
            raise SystemExit("--fixture-root is required for a continuation oracle")
        manifest = generate_continuation_oracle(
            args.fixture_root.resolve(), args.output.resolve()
        )
    else:
        if args.fixture_root is not None:
            raise SystemExit("--fixture-root requires --continuation-oracle")
        manifest = generate(args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
