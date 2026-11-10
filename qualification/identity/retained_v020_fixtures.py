"""Retained-family fixture constructors extracted unchanged from the frozen v0.2 corpus harness.

No retired runtime is imported or implemented here. S-24 keeps the original
benchmark immutable; this test-only subset uses current package entry points.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

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
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.configuration.legacy_periodic import translate_periodic
from pssolver.configuration.simulation import InvocationSpec
from pssolver.engine import Engine
from pssolver.functional import (
    build_channel_activity_functional_runtime,
    build_functional_runtime,
    channel_activity_functional_request,
    periodic_activity_functional_request,
)
from pssolver.functional.validation import PeriodicProductionHandle
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.io.legacy.periodic_workflow import (
    load_periodic_checkpoint as load_periodic_checkpoint,
)
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    VELOCITY_COMPONENTS,
    CompleteStressBerisEdwards,
)
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)
from pssolver.runtime.legacy_entry import restore_periodic_state
from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows.channel_beris_edwards import (
    _load_checkpoint,
)
from pssolver.workflows.plane_checkpoint import load_plane_checkpoint as load_plane_checkpoint
from pssolver.workflows.plane_checkpoint import restore_plane_checkpoint as restore_plane_checkpoint


# Explicit re-export used by historical_continuations.
load_channel_complete_stress_checkpoint = _load_checkpoint

CONTINUATION_TOLERANCE_EPS_MULTIPLIER = 8.0


NUMERICAL_BUILD_FINGERPRINT_SCHEMA_VERSION = 2


PERIODIC_SHAPE = (6, 6, 4)


CHANNEL_SHAPE = (7, 6, 6)


PLANE_SHAPE = (7, 7, 6)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _torch_build_settings(value: str) -> str:
    """Extract Torch's stable build-settings payload.

    Human-readable CUDA/CuDNN probe lines vary with device visibility and
    between pip and conda-forge builds.  The single ``Build settings:`` line
    is the build contract; explicit version fields below bind the remaining
    Torch identity without depending on the presentation of ``show()``.
    """

    settings = [
        line.split("Build settings:", 1)[1].strip()
        for line in value.splitlines()
        if "Build settings:" in line
    ]
    if len(settings) != 1 or not settings[0]:
        raise RuntimeError("torch configuration must contain exactly one Build settings line")
    return settings[0]


def _numpy_build_configuration() -> tuple[str, object]:
    """Return NumPy build metadata across the supported NumPy API range."""

    try:
        return "mode_dicts", np.__config__.show(mode="dicts")
    except TypeError:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            np.__config__.show()
        value = stream.getvalue().strip()
        if not value:
            raise RuntimeError("NumPy configuration output is empty")  # noqa: B904 (frozen oracle)
        return "show_text", value


def numerical_build_fingerprint() -> dict[str, object]:
    """Identify the numerical build that controls exact FFT continuation."""

    torch_build_settings = _torch_build_settings(torch.__config__.show())
    numpy_source, numpy_build = _numpy_build_configuration()
    return {
        "fingerprint_schema_version": (NUMERICAL_BUILD_FINGERPRINT_SCHEMA_VERSION),
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
            json.dumps(numpy_build, separators=(",", ":"), sort_keys=True).encode("utf-8")
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
    request = periodic_activity_functional_request(periodic_simulation(root).specification)
    return build_functional_runtime(request)


def build_channel_functional(root: Path):
    request = channel_activity_functional_request(channel_simulation(root).specification)
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
    translated = translate_periodic(run_spec.simulation, flavour="production", load_initial=True)
    engine = Engine(translated.plan, device="cpu", dtype=torch.float64)
    state = engine.initial_state(translated.initial_values)
    return run_spec, PeriodicProductionHandle(engine, state)


def restore_periodic_checkpoint(handle, checkpoint, *, run_spec):
    return restore_periodic_state(handle.engine, handle.state, checkpoint, run_spec=run_spec)


def advance_periodic(handle):
    handle.engine.step_(handle.state)


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
    coordinate = torch.arange(np.prod(PLANE_SHAPE), dtype=torch.float64).reshape(PLANE_SHAPE)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    initial = {
        name: coordinate.mul((index + 1) * 1.0e-3) for index, name in enumerate(Q_COMPONENTS)
    }
    solver, projector = build_legacy_plane_runtime(
        run_spec,
        device="cpu",
        initial_values=initial,
    )
    return run_spec, LegacyPlaneRuntimeAdapter(solver, projector)


def _field_records(adapter, components: tuple[str, ...]) -> dict[str, object]:
    records = {}
    for name in components:
        records[name] = _tensor_record(adapter.fields[name])
        records[f"{name}.hat"] = _tensor_record(adapter.fields[f"{name}.hat"])
    records["completed_steps"] = adapter.completed_steps
    records["integrator_step_count"] = int(adapter.solver.integrator.step_count)
    records["integrator_refresh_count"] = int(adapter.solver.integrator.refresh_count)
    return records


def functional_continuation_records(state) -> dict[str, object]:
    return {
        "q_physical": _tensor_record(state[0]),
        "q_spectral": _tensor_record(state[1]),
        "completed_steps": 2,
    }


def periodic_continuation_records(adapter) -> dict[str, object]:
    records = {}
    for name in Q_COMPONENTS:
        records[name] = _tensor_record(adapter.fields[name])
        records[f"{name}.hat"] = _tensor_record(adapter.fields[f"{name}.hat"])
    records["completed_steps"] = adapter.completed_steps
    records["integrator_step_count"] = int(adapter.state.refresh_phase[0])
    records["integrator_refresh_count"] = int(adapter.state.refresh_phase[1])
    return records


def channel_complete_continuation_records(adapter) -> dict[str, object]:
    records = _field_records(
        adapter,
        (*tuple(Q_COMPONENTS), *tuple(VELOCITY_COMPONENTS), "p"),
    )
    records["pressure_guess"] = _tensor_record(adapter.capture_pressure_guess())
    return records


def plane_continuation_records(adapter) -> dict[str, object]:
    return _field_records(adapter, tuple(Q_COMPONENTS))


def _real_dtype(dtype: np.dtype) -> np.dtype:
    return np.empty((), dtype=dtype).real.dtype


def _compare_continuation(
    actual_records: dict[str, object],
    actual_arrays: dict[str, np.ndarray],
    expected_records: dict[str, object],
    oracle_root: Path,
    case: str,
) -> dict[str, object]:
    oracle_manifest = json.loads((oracle_root / "manifest.json").read_text(encoding="utf-8"))
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
            raise AssertionError(f"matching-build continuation digest differs for {name}")
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
