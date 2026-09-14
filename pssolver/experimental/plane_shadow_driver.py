"""Explicit CPU shadow entry point for a completed production Plane run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

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
from pssolver.execution import (
    AlgebraicExecutionPolicy,
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
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

from ._shadow_support import file_sha256, ordered_tensor_sha256
from .beris_edwards import (
    PlaneBerisEdwardsSolverOptions,
    create_beris_edwards_plane_geometry_solver_registry,
)
from .model_execution import (
    ExperimentalModelRuntime,
    build_experimental_model_runtime,
)
from .shadow_metadata import (
    ShadowMetadataComparison,
    compare_shadow_to_production_metadata,
    plane_beris_edwards_production_signature,
)
from .shadow_run import ExperimentalPlaneShadowRun
from .stokes import PlaneStokesSolverOptions


_PRODUCTION_SCRIPT = "Plane_beris_edwards_stokes.py"
_PRODUCTION_MODEL = "beris_edwards_complete_nematic_stress_stokes"


def _load_json_mapping(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object")
    return dict(value)


def _boundary_set(values: object, description: str) -> BoundarySet:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{description} must contain three boundary names")
    conditions = {
        "periodic": PeriodicBC,
        "neumann": HomogeneousNeumannBC,
        "dirichlet": HomogeneousDirichletBC,
    }
    try:
        return BoundarySet(tuple(conditions[value]() for value in values))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{description} contains an unsupported boundary") from exc


def _require_common_shadow_production_metadata(
    metadata: Mapping[str, object],
) -> None:
    """Validate the production contract shared by shadow qualifications."""

    try:
        if metadata["script"] != _PRODUCTION_SCRIPT:
            raise ValueError("production script identity is incompatible")
        if metadata["status"] != "complete":
            raise ValueError("production reference metadata is not complete")
        model = metadata["model"]
        if model["name"] != "active_nematics" or model["variant"] != (
            _PRODUCTION_MODEL
        ):
            raise ValueError("production model identity is incompatible")
        if metadata["solver"]["real_dtype"] != "float64":
            raise ValueError("shadow qualification requires float64")
        if metadata["save_start_step"] != 0:
            raise ValueError("shadow qualification requires save_start_step=0")
        if metadata["save_hydrodynamics"] is not True:
            raise ValueError("shadow qualification requires saved u and p")
        if metadata["numerics"]["q_gradient_reuse"]["enabled"] is not True:
            raise ValueError("shadow qualification requires Q-gradient reuse")
        if metadata["numerics"]["precision"]["tf32_effective"] is not False:
            raise ValueError("shadow qualification requires TF32 to be ineffective")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "production metadata lacks the shadow reference contract"
        ) from exc


def _require_stage_l_production_metadata(
    metadata: Mapping[str, object],
) -> None:
    """Restrict the first shadow gate to completed CPU/float64 references."""

    _require_common_shadow_production_metadata(metadata)
    try:
        if metadata["runtime_environment"]["device_type"] != "cpu":
            raise ValueError("Stage L requires a CPU production reference")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "production metadata lacks the Stage L device contract"
        ) from exc


def _require_stage_m_h100_production_metadata(
    metadata: Mapping[str, object],
    *,
    expected_gpu_name: str,
) -> None:
    """Restrict Stage M to an explicit completed H100/float64 reference."""

    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")
    _require_common_shadow_production_metadata(metadata)
    try:
        runtime = metadata["runtime_environment"]
        if runtime["device_type"] != "cuda":
            raise ValueError("Stage M requires a CUDA production reference")
        reference_name = runtime["cuda_device_name"]
        if (
            not isinstance(reference_name, str)
            or expected_gpu_name.lower() not in reference_name.lower()
        ):
            raise ValueError(
                "Stage M production GPU does not match expected_gpu_name"
            )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "production metadata lacks the Stage M H100 device contract"
        ) from exc


@dataclass(frozen=True, slots=True)
class ProductionPlaneReference:
    """Validated configuration and initial Q from one production run."""

    directory: Path
    metadata: Mapping[str, object]
    metadata_sha256: str
    initial_q_path: Path
    initial_q_file_sha256: str
    initial_values: Mapping[str, torch.Tensor]


def _load_production_plane_reference(
    directory: str | Path,
    *,
    metadata_validator,
) -> ProductionPlaneReference:
    """Load one validated production reference without mutating it."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"production reference directory is missing: {directory}"
        )
    complete = directory / "COMPLETE"
    if not complete.is_file() or complete.read_text(encoding="utf-8") != (
        "complete\n"
    ):
        raise ValueError("production reference lacks a valid COMPLETE marker")
    metadata_path = directory / "metadata.json"
    metadata_sha256 = file_sha256(metadata_path)
    metadata = _load_json_mapping(metadata_path, "production metadata")
    if file_sha256(metadata_path) != metadata_sha256:
        raise RuntimeError("production metadata changed while it was read")
    metadata_validator(metadata)

    signature = plane_beris_edwards_production_signature(metadata)
    shape = tuple(signature["solver"]["shape"])
    initial_q_path = directory / "Q_0.npy"
    if not initial_q_path.is_file():
        raise FileNotFoundError(
            f"production initial Q snapshot is missing: {initial_q_path}"
        )
    initial_q_file_sha256 = file_sha256(initial_q_path)
    q = np.load(initial_q_path, allow_pickle=False)
    if file_sha256(initial_q_path) != initial_q_file_sha256:
        raise RuntimeError("production Q_0 changed while it was read")
    if q.shape != (*shape, len(Q_COMPONENTS)):
        raise ValueError("production Q_0 has the wrong shape")
    if q.dtype != np.dtype(signature["solver"]["real_dtype"]):
        raise ValueError("production Q_0 has the wrong dtype")
    if not np.isfinite(q).all():
        raise ValueError("production Q_0 contains NaN or Inf")
    initial_values = {
        name: torch.from_numpy(np.array(q[..., index], copy=True))
        for index, name in enumerate(Q_COMPONENTS)
    }
    # Production hashes the in-solver representation, including batch size 1,
    # while its Q_0 file deliberately omits that leading storage dimension.
    observed_q_sha256 = ordered_tensor_sha256(
        {
            name: value.unsqueeze(0)
            for name, value in initial_values.items()
        }
    )
    try:
        expected_q_sha256 = metadata["initial_condition"][
            "projected_q_sha256"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "production metadata lacks projected initial-Q identity"
        ) from exc
    if observed_q_sha256 != expected_q_sha256:
        raise ValueError("production Q_0 identity differs from metadata")

    return ProductionPlaneReference(
        directory=directory,
        metadata=metadata,
        metadata_sha256=metadata_sha256,
        initial_q_path=initial_q_path,
        initial_q_file_sha256=initial_q_file_sha256,
        initial_values=initial_values,
    )


def load_production_plane_reference(
    directory: str | Path,
) -> ProductionPlaneReference:
    """Load a completed Stage L CPU/float64 production reference."""

    return _load_production_plane_reference(
        directory,
        metadata_validator=_require_stage_l_production_metadata,
    )


def load_h100_production_plane_reference(
    directory: str | Path,
    *,
    expected_gpu_name: str = "H100",
) -> ProductionPlaneReference:
    """Load a completed Stage M H100/float64 production reference."""

    return _load_production_plane_reference(
        directory,
        metadata_validator=lambda metadata: (
            _require_stage_m_h100_production_metadata(
                metadata,
                expected_gpu_name=expected_gpu_name,
            )
        ),
    )


def _build_plane_shadow_runtime_from_production_metadata(
    metadata: Mapping[str, object],
    *,
    device: str | torch.device,
    enable_performance_instrumentation: bool = False,
    algebraic_execution_policy: AlgebraicExecutionPolicy | None = None,
    enable_algebraic_representation_reuse: bool | None = None,
    enable_lazy_algebraic_materialization: bool | None = None,
    enable_batched_physical_islands: bool | None = None,
) -> tuple[ExperimentalModelRuntime, ShadowMetadataComparison]:
    """Construct and parity-check a migrated runtime after contract checks."""

    device = torch.device(device)
    signature = plane_beris_edwards_production_signature(metadata)
    solver = signature["solver"]
    parameters = signature["model"]["parameters"]
    boundaries = signature["boundary_conditions"]
    numerical = signature["numerics"]

    q_boundaries = _boundary_set(boundaries["Q"], "Q boundaries")
    tangential_boundaries = _boundary_set(
        boundaries["velocity_tangential"],
        "tangential velocity boundaries",
    )
    normal_boundaries = _boundary_set(
        boundaries["velocity_normal"],
        "normal velocity boundaries",
    )
    pressure_boundaries = _boundary_set(
        boundaries["pressure"],
        "pressure boundaries",
    )
    zero_mode = TangentialZeroModePolicy(numerical["velocity_zero_mode"])
    constitutive = BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=q_boundaries,
        tangential_boundaries=tangential_boundaries,
        normal_boundaries=normal_boundaries,
        pressure_boundaries=pressure_boundaries,
        parameters=BerisEdwardsConstitutiveParameters(
            ldg_a=parameters["ldg_A"],
            ldg_b=parameters["ldg_B"],
            ldg_c=parameters["ldg_C"],
            ldg_l1=parameters["ldg_L1"],
            flow_alignment=parameters["flow_alignment_lambda"],
            active_prefactor=parameters["active_stress_prefactor"],
        ),
        stokes_system=IncompressibleStokesSystemSpec(
            name="flow",
            force_components=NEMATIC_FORCE_COMPONENTS,
            velocity_components=VELOCITY_COMPONENTS,
            pressure_component="p",
            viscosity=parameters["eta"],
            friction=parameters["fric"],
            tangential_zero_mode_policy=zero_mode,
        ),
        initial_amplitude=0.0,
    )
    model = BerisEdwardsPlaneCoupledModel(
        constitutive_model=constitutive,
        rotational_viscosity=parameters["rotational_viscosity_gamma"],
    )
    numerics = NumericsConfig(
        precision=Precision(solver["real_dtype"]),
        dealias_rule=DealiasRule(numerical["dealias_rule"]),
        transform_execution_order=TransformExecutionOrder(
            solver["transform_execution_order"]
        ),
        projected_transform_execution=ProjectedTransformExecution(
            numerical["projected_transform_execution"]
        ),
        spectral_storage=SpectralStorage(solver["spectral_storage"]),
        hermitian_axis=numerical["hermitian_axis"],
    )
    pressure_diagnostics = bool(
        metadata["numerics"]["pressure_residual_diagnostics"]
    )
    runtime = build_experimental_model_runtime(
        model,
        PlaneSlab(
            DomainSpec(
                tuple(solver["shape"]),
                tuple(solver["lengths"]),
            )
        ),
        numerics,
        dt=solver["dt"],
        device=device,
        batch_size=1,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space=numerical[
                        "molecular_field_linear_space"
                    ],
                    stress_divergence_sum_space=numerical[
                        "stress_divergence_sum_space"
                    ],
                    pointwise_execution=numerical["pointwise_execution"],
                ),
                plane_stokes_options=PlaneStokesSolverOptions(
                    pressure_diagnostics=pressure_diagnostics
                ),
            )
        ),
        enable_performance_instrumentation=(
            enable_performance_instrumentation
        ),
        algebraic_execution_policy=algebraic_execution_policy,
        enable_algebraic_representation_reuse=(
            enable_algebraic_representation_reuse
        ),
        enable_lazy_algebraic_materialization=(
            enable_lazy_algebraic_materialization
        ),
        enable_batched_physical_islands=enable_batched_physical_islands,
    )
    runtime.solver.integrator.set_spectral_refresh_interval(
        numerical["spectral_refresh_interval_steps"]
    )
    comparison = compare_shadow_to_production_metadata(runtime, metadata)
    comparison.require_compatible()
    return runtime, comparison


def build_plane_shadow_runtime_from_production_metadata(
    metadata: Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
    algebraic_execution_policy: AlgebraicExecutionPolicy | None = None,
    enable_algebraic_representation_reuse: bool | None = None,
    enable_lazy_algebraic_materialization: bool | None = None,
    enable_batched_physical_islands: bool | None = None,
) -> tuple[ExperimentalModelRuntime, ShadowMetadataComparison]:
    """Construct the Stage L CPU shadow runtime from production metadata."""

    _require_stage_l_production_metadata(metadata)
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("Stage L shadow qualification is CPU-only")
    return _build_plane_shadow_runtime_from_production_metadata(
        metadata,
        device=device,
        algebraic_execution_policy=algebraic_execution_policy,
        enable_algebraic_representation_reuse=(
            enable_algebraic_representation_reuse
        ),
        enable_lazy_algebraic_materialization=(
            enable_lazy_algebraic_materialization
        ),
        enable_batched_physical_islands=enable_batched_physical_islands,
    )


def build_h100_plane_shadow_runtime_from_production_metadata(
    metadata: Mapping[str, object],
    *,
    expected_gpu_name: str = "H100",
    device: str | torch.device = "cuda",
    enable_performance_instrumentation: bool = False,
    algebraic_execution_policy: AlgebraicExecutionPolicy | None = None,
    enable_algebraic_representation_reuse: bool | None = None,
    enable_lazy_algebraic_materialization: bool | None = None,
    enable_batched_physical_islands: bool | None = None,
) -> tuple[ExperimentalModelRuntime, ShadowMetadataComparison]:
    """Construct the opt-in Stage M runtime on the expected H100 device."""

    _require_stage_m_h100_production_metadata(
        metadata,
        expected_gpu_name=expected_gpu_name,
    )
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Stage M shadow qualification requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage M requested CUDA but torch.cuda.is_available() is false"
        )
    actual_name = torch.cuda.get_device_name(device)
    if expected_gpu_name.lower() not in actual_name.lower():
        raise RuntimeError(
            "Stage M execution GPU does not match expected_gpu_name: "
            f"{actual_name!r}"
        )
    return _build_plane_shadow_runtime_from_production_metadata(
        metadata,
        device=device,
        enable_performance_instrumentation=(
            enable_performance_instrumentation
        ),
        algebraic_execution_policy=algebraic_execution_policy,
        enable_algebraic_representation_reuse=(
            enable_algebraic_representation_reuse
        ),
        enable_lazy_algebraic_materialization=(
            enable_lazy_algebraic_materialization
        ),
        enable_batched_physical_islands=enable_batched_physical_islands,
    )


def run_plane_shadow_from_production_reference(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    confirmed_steps: int,
) -> dict[str, object]:
    """Replay one complete short production run through the shadow runtime."""

    if (
        not isinstance(confirmed_steps, int)
        or isinstance(confirmed_steps, bool)
        or confirmed_steps <= 0
    ):
        raise ValueError("confirmed_steps must be a positive integer")
    reference = load_production_plane_reference(production_directory)
    completed_steps = reference.metadata.get("completed_steps")
    if completed_steps != confirmed_steps:
        raise ValueError(
            "--confirm-steps must exactly equal the production completed_steps"
        )
    save_interval = reference.metadata["solver"]["save_interval"]
    if (
        not isinstance(save_interval, int)
        or isinstance(save_interval, bool)
        or save_interval <= 0
    ):
        raise ValueError("production save_interval must be positive")

    runtime, comparison = build_plane_shadow_runtime_from_production_metadata(
        reference.metadata,
        device="cpu",
    )
    initial_metadata = {
        "name": "production_projected_q_snapshot",
        "source_run_directory": str(reference.directory),
        "source_step": 0,
        "source_metadata_sha256": reference.metadata_sha256,
        "source_q_file": reference.initial_q_path.name,
        "source_q_file_sha256": reference.initial_q_file_sha256,
        "source_projected_q_sha256": reference.metadata[
            "initial_condition"
        ]["projected_q_sha256"],
        "configuration_comparison": comparison.to_metadata(),
    }
    run = ExperimentalPlaneShadowRun(
        runtime,
        output_directory,
        initial_values=reference.initial_values,
        initial_condition_metadata=initial_metadata,
    )
    saved_steps = [run.save_observation().step]
    for step in range(1, confirmed_steps + 1):
        run.advance(1)
        if step % save_interval == 0 or step == confirmed_steps:
            saved_steps.append(run.save_observation().step)
    final = run.complete()
    return {
        "classification": "shadow_run_complete",
        "production_directory": str(reference.directory),
        "shadow_directory": str(run.output_directory),
        "completed_steps": final.step,
        "saved_steps": saved_steps,
        "configuration_compatible": comparison.compatible,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in Stage L Plane Beris-Edwards shadow path from a "
            "completed CPU/float64 production reference."
        )
    )
    parser.add_argument(
        "--production-reference-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm-steps",
        type=int,
        required=True,
        help=(
            "Explicit safety confirmation; it must equal the completed step "
            "count recorded by the production reference."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_plane_shadow_from_production_reference(
        args.production_reference_dir,
        args.output_dir,
        confirmed_steps=args.confirm_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ProductionPlaneReference",
    "build_h100_plane_shadow_runtime_from_production_metadata",
    "build_plane_shadow_runtime_from_production_metadata",
    "load_h100_production_plane_reference",
    "load_production_plane_reference",
    "main",
    "run_plane_shadow_from_production_reference",
]
