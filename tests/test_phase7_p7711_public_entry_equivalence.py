"""P7.7.11 public-entry byte identity and same-runtime restart gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pytest
import torch

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SimulationResult,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.applications.channel_active_nematics import (
    run_channel_active_nematics,
)
from pssolver.applications.plane_beris_edwards import (
    run_plane_beris_edwards,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PUBLIC_PLANE_APPLICATION,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import CompleteStressBerisEdwards


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p7711_cpu_equivalence.json"
)
CASES = (
    ("plane", "legacy_production"),
    ("plane", "compiled_v2"),
    ("channel", "legacy_channel"),
    ("channel", "compiled_channel_v2"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle_environment() -> dict[str, object]:
    """Identify the software/build scope of a byte-level trajectory oracle."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
    }


def _assert_environment_scoped_oracle(
    observed: dict[str, str],
    record: dict[str, object],
    key: str,
    *,
    environment: dict[str, object],
) -> None:
    oracle = record["continuous_oracle_contract"]
    if environment == oracle["reference_environment"]:
        assert observed == record["continuous_oracle_sha256"][key]
        return
    # Raw NPY hashes are implementation fingerprints, not portable numerical
    # tolerances. The caller must establish all same-environment byte-identity
    # gates before presenting this observed fingerprint.
    assert set(observed) == {
        "Q_2.npy",
        "u_2.npy",
        "p_2.npy",
        "diagnostics.npy",
    }
    assert all(
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
        for value in observed.values()
    )


def _plane_simulation(
    output: Path,
    runtime_path: str,
    *,
    steps: int,
    checkpoint_interval: int | None = None,
    restart_from: Path | None = None,
) -> Simulation:
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
    geometry = PlaneSlab(
        shape=(8, 8, 6),
        lengths=(100.0, 100.0, 20.0),
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
        ),
        time=TimeStepping(
            dt=0.005,
            refresh={"mode": "disabled"},
        ),
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
            runtime_path=runtime_path,
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
            directory=output,
            steps=steps,
            save_interval=100,
            diagnostic_interval=100,
            checkpoint_interval=checkpoint_interval,
            restart_from=restart_from,
        ),
        invocation=InvocationSpec({}),
    )


def _channel_simulation(
    output: Path,
    runtime_path: str,
    *,
    steps: int,
    checkpoint_interval: int | None = None,
    restart_from: Path | None = None,
) -> Simulation:
    request = ChannelActiveNematicRunSpec(
        shape=(8, 6, 5),
        lengths=(8.0, 6.0, 5.0),
        dt=1.0e-3,
        steps=steps,
        save_interval=100,
        diagnostic_interval=100,
        checkpoint_interval=checkpoint_interval,
        restart_from=restart_from,
        generated_output_directory=output,
        snapshot_output_directory=output.with_name(f"{output.name}_snapshot"),
        device="cpu",
        dtype="float32",
        activity=0.2,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=100,
        pressure_fixed_iterations=8,
        runtime_path=runtime_path,
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
        invocation=source.invocation,
    )


def _simulation(
    application: str,
    output: Path,
    runtime_path: str,
    *,
    steps: int,
    checkpoint_interval: int | None = None,
    restart_from: Path | None = None,
) -> Simulation:
    factory = _plane_simulation if application == "plane" else _channel_simulation
    return factory(
        output,
        runtime_path,
        steps=steps,
        checkpoint_interval=checkpoint_interval,
        restart_from=restart_from,
    )


def _run_direct(simulation: Simulation):
    compiled = compile_simulation(simulation)
    request = compiled.application_request
    if compiled.application == PUBLIC_PLANE_APPLICATION:
        result = run_plane_beris_edwards(request)
    elif compiled.application == PUBLIC_CHANNEL_APPLICATION:
        result = run_channel_active_nematics(request)
    else:  # pragma: no cover - protected by the public compiler registry.
        raise AssertionError("unexpected qualified application")
    return compiled, result


def _run_public(simulation: Simulation):
    compiled = compile_simulation(simulation)
    return compiled, run_simulation(compiled)


def _array_file_names(directory: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in directory.glob("*.npy")
            if path.name.startswith(("Q_", "u_", "p_"))
        )
    )


def _assert_array_artifacts_identical(left: Path, right: Path) -> None:
    names = _array_file_names(left)
    assert names == _array_file_names(right)
    assert names
    for name in names:
        left_path = left / name
        right_path = right / name
        assert left_path.read_bytes() == right_path.read_bytes(), name
        left_array = np.load(left_path, allow_pickle=False)
        right_array = np.load(right_path, allow_pickle=False)
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert np.array_equal(left_array, right_array), name
    for name in ("diagnostics.npy", "diagnostics.csv", "COMPLETE"):
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def _assert_workflow_metadata_agrees(
    left: Path,
    right: Path,
    *,
    runtime_path: str,
    final_step: int,
) -> None:
    left_metadata = json.loads((left / "metadata.json").read_text())
    right_metadata = json.loads((right / "metadata.json").read_text())
    for metadata in (left_metadata, right_metadata):
        assert metadata["status"] == "complete"
        assert metadata["completed_steps"] == final_step
        assert metadata["runtime_selection"]["requested"] == runtime_path
        assert metadata["runtime_selection"]["effective"] == runtime_path
        assert metadata["runtime_selection"]["fallback_used"] is False
    for key in ("runtime_selection", "boundary_conditions", "numerics"):
        assert left_metadata[key] == right_metadata[key]
    assert left_metadata["workflow"] == right_metadata["workflow"]


def _assert_raw_and_public_result_agree(raw, public: SimulationResult) -> None:
    assert public.completed is True
    assert public.start_step == raw.start_step
    assert public.final_step == raw.final_step
    assert public.saved_steps == raw.saved_steps
    assert public.checkpoint_steps == raw.checkpoint_steps
    for name in ("q", "velocity", "pressure"):
        left = getattr(raw.final_observation, name)
        right = getattr(public.final_observation, name)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert np.array_equal(left, right)
    assert tuple(value.as_tuple() for value in raw.diagnostics) == tuple(
        value.as_tuple() for value in public.diagnostics
    )


def _final_oracle(directory: Path, final_step: int) -> dict[str, str]:
    return {
        name: _sha256(directory / name)
        for name in (
            f"Q_{final_step}.npy",
            f"u_{final_step}.npy",
            f"p_{final_step}.npy",
            "diagnostics.npy",
        )
    }


def _exercise_case(
    tmp_path: Path,
    application: str,
    runtime_path: str,
) -> dict[str, str]:
    root = tmp_path / f"{application}_{runtime_path}"
    direct_directory = root / "direct_continuous"
    public_directory = root / "public_continuous"

    direct_compiled, direct = _run_direct(
        _simulation(
            application,
            direct_directory,
            runtime_path,
            steps=2,
        )
    )
    public_compiled, public = _run_public(
        _simulation(
            application,
            public_directory,
            runtime_path,
            steps=2,
        )
    )
    assert direct_compiled.application == public_compiled.application
    assert direct_compiled.application_request.runtime_identity_sha256() == (
        public_compiled.application_request.runtime_identity_sha256()
    )
    _assert_raw_and_public_result_agree(direct, public)
    _assert_array_artifacts_identical(direct_directory, public_directory)
    _assert_workflow_metadata_agrees(
        direct_directory,
        public_directory,
        runtime_path=runtime_path,
        final_step=2,
    )
    assert public.output_directory == public_directory.resolve()
    assert public.to_metadata()["final_observation"]["step"] == 2

    segment_directory = root / "public_segment"
    _, segment = _run_public(
        _simulation(
            application,
            segment_directory,
            runtime_path,
            steps=1,
            checkpoint_interval=1,
        )
    )
    checkpoint = segment_directory / "checkpoint_1"
    assert segment.checkpoint_steps == (1,)
    assert checkpoint.is_dir()

    direct_resume_directory = root / "direct_resume"
    public_resume_directory = root / "public_resume"
    direct_resume_compiled, direct_resume = _run_direct(
        _simulation(
            application,
            direct_resume_directory,
            runtime_path,
            steps=1,
            restart_from=checkpoint,
        )
    )
    public_resume_compiled, public_resume = _run_public(
        _simulation(
            application,
            public_resume_directory,
            runtime_path,
            steps=1,
            restart_from=checkpoint,
        )
    )
    assert direct_resume_compiled.application_request.runtime_identity_sha256() == (
        public_resume_compiled.application_request.runtime_identity_sha256()
    )
    _assert_raw_and_public_result_agree(direct_resume, public_resume)
    _assert_array_artifacts_identical(
        direct_resume_directory,
        public_resume_directory,
    )
    _assert_workflow_metadata_agrees(
        direct_resume_directory,
        public_resume_directory,
        runtime_path=runtime_path,
        final_step=2,
    )
    for resumed in (direct_resume_directory, public_resume_directory):
        for name in ("Q_2.npy", "u_2.npy", "p_2.npy"):
            assert (resumed / name).read_bytes() == (
                public_directory / name
            ).read_bytes()
        metadata = json.loads((resumed / "metadata.json").read_text())
        assert metadata["restart"]["source_completed_steps"] == 1
        assert metadata["restart"]["source_runtime_path"] == runtime_path
    assert public_resume.start_step == 1
    assert public_resume.final_step == 2
    return _final_oracle(direct_directory, 2)


@pytest.mark.parametrize(("application", "runtime_path"), CASES)
def test_old_and_public_entries_are_byte_identical_with_restart(
    tmp_path,
    application,
    runtime_path,
):
    record = json.loads(RESULT_PATH.read_text())
    observed = _exercise_case(tmp_path, application, runtime_path)
    key = f"{application}:{runtime_path}"
    _assert_environment_scoped_oracle(
        observed,
        record,
        key,
        environment=_oracle_environment(),
    )


def test_absolute_oracle_is_scoped_to_its_recorded_software_build():
    record = json.loads(RESULT_PATH.read_text())
    key = "plane:legacy_production"
    observed = record["continuous_oracle_sha256"][key]
    _assert_environment_scoped_oracle(
        observed,
        record,
        key,
        environment={"different_supported_build": True},
    )
    changed = {**observed, "Q_2.npy": "0" * 64}
    with pytest.raises(AssertionError):
        _assert_environment_scoped_oracle(
            changed,
            record,
            key,
            environment=record["continuous_oracle_contract"][
                "reference_environment"
            ],
        )


def test_machine_record_freezes_p7711_scope_and_prior_implementation():
    record = json.loads(RESULT_PATH.read_text())
    assert record["classification"] == (
        "PASS_P7_7_11_CPU_BYTE_IDENTITY_AND_RESTART"
    )
    assert record["qualified_cases"] == [
        f"{application}:{runtime_path}"
        for application, runtime_path in CASES
    ]
    assert record["direct_and_public_byte_identity"] is True
    assert record["same_runtime_restart_byte_identity"] is True
    assert record["continuous_oracle_contract"]["scope"] == (
        "implementation_fingerprint_for_matching_environment"
    )
    assert record["continuous_oracle_contract"][
        "cross_environment_raw_sha_required"
    ] is False
    assert record["continuous_oracle_contract"][
        "same_environment_byte_identity_required"
    ] is True
    assert record["p7_7_12_complete"] is False
    assert record["phase_8_authorized"] is False
    assert record["qualified_source_sha256"] == {
        "pssolver/api/results.py": (
            "8ed89056f2c3b53c94d9947fda8fb048ce26a40636f9e732181f5e2f63b262a2"
        ),
        "pssolver/api/runner.py": (
            "ccd06a71a0c5731ca32d54683f4f6220bd67b27a6bba52f433abd372ea8527eb"
        ),
        "pssolver/runtime/plane_legacy.py": (
            "4cc51416927bf544edda7a4a7c07fa415ca3b392d56020e574b8a89bce0eb295"
        ),
    }
