"""Temporary v0.4 declaration entrance to Engine and Runner; retired at S-28."""

from __future__ import annotations

# Plane's frozen declaration entrance; time integration is owned by Runner.
import hashlib
import json
import platform
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.execution_device import (
    configure_tf32_execution,
    execution_device_normalization,
    resolve_execution_device,
    scoped_tf32_execution,
)
from pssolver.configuration.legacy_periodic import (
    PeriodicBerisEdwardsRunSpec,
    _reject_restart,
    load_periodic_initial_q,
    preflight_periodic_restart,
    translate_periodic,
    validate_periodic_restore,
)
from pssolver.configuration.legacy_plane import (
    PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    load_plane_checkpoint,
    plane_backend_restart,
    plane_run_spec_dynamics_identity,
    read_plane_checkpoint_header,
    translate_plane,
    validate_plane_restore,
)
from pssolver.configuration.plane_beris_edwards_component_graph import (
    PlaneBerisEdwardsRunComponents,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation import SimulationSpec
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.engine.engine import Engine
from pssolver.engine.progress import (
    _refresh_counters_are_consistent,
    _refresh_counters_from_completed_steps,
)
from pssolver.engine.state import State
from pssolver.errors import SimulationRestartError
from pssolver.io import engine_state
from pssolver.io.checkpoint import CheckpointError
from pssolver.io.legacy.periodic_workflow import load_periodic_checkpoint
from pssolver.models.active_nematics import (
    BerisEdwardsPointwiseKernels,
    Q_convention_metadata,
    create_initial_condition,
    sample_periodic_neutral_defects_2d,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.operators.projection import DEALIAS_RULE_FRACTIONS
from pssolver.planning import TransformKind

from .observations import PeriodicCapture
from .plane_observations import PlaneCapture
from .runner import Runner, RunResult, RunSchedule


def restore_periodic_state(engine, state, checkpoint, *, run_spec):
    """Validate and materialize all legacy Q tensors before mutating caller state."""
    fields = engine.field_view(state)
    spatial = {name: fields[name] for name in Q_COMPONENTS}
    spectral = {name: fields[f"{name}.hat"] for name in Q_COMPONENTS}
    validate_periodic_restore(
        checkpoint,
        run_spec=run_spec,
        target_spatial=spatial,
        target_spectral=spectral,
        target_refresh_interval=engine.plan.integrator.refresh_interval,
        backend_restart={"kind": "periodic_stokes_stateless", "state_keys": []},
    )
    pending = tuple(
        (target[name], source[name].to(device=engine.device))
        for name in Q_COMPONENTS
        for target, source in (
            (spatial, checkpoint.evolved_spatial),
            (spectral, checkpoint.evolved_spectral),
        )
    )
    for target, source in pending:
        target.copy_(source)
    state.spatial[len(Q_COMPONENTS) :].zero_()
    state.spectral[len(Q_COMPONENTS) :].zero_()
    state.constrained_current = False
    engine.prepare_constraints_(state)
    state.completed_steps = checkpoint.completed_steps
    state.refresh_phase = (
        checkpoint.integrator_step_count,
        checkpoint.integrator_refresh_count,
    )
    if not _refresh_counters_are_consistent(
        completed_steps=state.completed_steps,
        refresh_interval=engine.plan.integrator.refresh_interval,
        refresh_step_count=state.refresh_phase[0],
        refresh_count=state.refresh_phase[1],
    ):
        raise ValueError("restored refresh counters are inconsistent")
    return state.completed_steps


def _run_v04_translation_owned(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    tf32_metadata: dict[str, object],
    public_compilation_metadata: dict[str, object] | None = None,
):
    """Execute one frozen P8.2 application with no runtime fallback."""

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
    spec = run_spec.simulation
    stokes = spec.equation_system.parameters["stokes"]["parameters"]
    uniform_mode_action = (
        "remove_all_uniform_velocity_and_force"
        if stokes["tangential_zero_mode_policy"] == "zero_mean"
        else "retain_all_uniform_velocity_resolved_by_friction"
    )
    metadata = {
        "schema_version": 1,
        "application": "periodic_complete_stress_beris_edwards",
        "configuration": spec.to_metadata(),
        "configuration_sha256": spec.canonical_sha256(),
        "runtime_identity_sha256": run_spec.runtime_identity_sha256(),
        "runtime_selection": {
            "requested": run_spec.runtime_path,
            "effective": run_spec.runtime_path,
            "fallback_allowed": False,
            "fallback_used": False,
        },
        "normalization": execution_device_normalization(spec.execution.options["device"]),
        "pressure_gauge": "zero_mean",
        "tangential_zero_mode_policy": stokes["tangential_zero_mode_policy"],
        "friction": stokes["friction"],
        "uniform_velocity_mode_action": uniform_mode_action,
    }
    if public_compilation_metadata is not None:
        metadata["public_compilation"] = dict(public_compilation_metadata)
    if emit_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    if run_spec.dry_run:
        return None

    device = resolve_execution_device(spec.execution.options["device"])
    real_dtype = getattr(torch, spec.numerics.precision.value)
    metadata["precision"] = {
        "real_dtype": spec.numerics.precision.value,
        "spectral_dtype": ("complex64" if real_dtype == torch.float32 else "complex128"),
        **tf32_metadata,
    }
    translation = translate_periodic(spec, flavour="production", load_initial=False)
    restart = spec.workflow.options["restart_from"]
    restart_kind = None
    start = 0
    if restart is not None:
        try:
            restart_kind, header = preflight_periodic_restart(
                restart, run_spec=run_spec, plan=translation.plan, dtype=real_dtype
            )
            start = (
                header["completed_steps"]
                if restart_kind == "engine_state"
                else header.completed_steps
            )
        except CheckpointError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            _reject_restart(
                str(exc) or "periodic checkpoint preflight failed",
                reason="checkpoint_preflight_failed",
                details={"legacy_exception_type": type(exc).__name__},
                exception=exc,
            )
        if spec.workflow.options["save_start_step"] > start + spec.workflow.steps:
            _reject_restart(
                "save_start_step must not exceed the absolute final step",
                reason="save_start_step_after_final_step",
                step=start,
                details={
                    "save_start_step": spec.workflow.options["save_start_step"],
                    "absolute_final_step": start + spec.workflow.steps,
                },
            )
    initial_values, snapshot_path, snapshot_sha256 = load_periodic_initial_q(run_spec)
    metadata["initial_condition"] = {
        "source": str(snapshot_path),
        "sha256": snapshot_sha256,
        "logical_mode": spec.initial_condition.parameters["mode"],
    }
    output = run_spec.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty periodic output: {output}")
    engine = Engine(translation.plan, device=device, dtype=real_dtype)
    state = engine.initial_state(
        {name: value.to(device=device) for name, value in initial_values.items()}
    )
    metadata["runtime_construction"] = {
        "plan_identity_sha256": engine.plan.identity_sha256,
        "engine": "pssolver.engine.engine.Engine",
        "operation_order": [
            "prepare_constraints",
            "explicit",
            "euler",
            "project",
            "inverse",
        ],
    }
    metadata["runtime"] = {
        "requested": run_spec.runtime_path,
        "effective": run_spec.runtime_path,
        "adapter": "pssolver.engine.engine.Engine",
        "fallback_used": False,
        "pressure_gauge": "zero_mean",
        "hermitian_state_projection": engine.projector.execution_metadata()[
            "hermitian_state_projection"
        ],
        "tangential_zero_mode_policy": stokes["tangential_zero_mode_policy"],
        "friction": float(stokes["friction"]),
        "uniform_velocity_mode_action": uniform_mode_action,
    }
    if restart is not None:
        try:
            if restart_kind == "engine_state":
                state = engine_state.read(
                    restart,
                    plan_identity_sha256=engine.plan.identity_sha256,
                    dtype=real_dtype,
                    batch=1,
                    device=device,
                )
            else:
                restore_periodic_state(
                    engine, state, load_periodic_checkpoint(restart), run_spec=run_spec
                )
        except (SimulationRestartError, CheckpointError):
            raise
        except Exception as exc:
            raise SimulationRestartError(
                str(exc) or "periodic checkpoint restore failed",
                application_family="periodic_complete_stress_beris_edwards",
                stage="restart_restore",
                step=state.completed_steps,
                details={
                    "reason": "checkpoint_restore_failed",
                    "legacy_exception_type": type(exc).__name__,
                },
            ) from exc
    options = spec.workflow.options
    workflow = Runner(
        engine,
        schedule=RunSchedule(
            steps=spec.workflow.steps,
            save_interval=options["save_interval"],
            save_start_step=options["save_start_step"],
            diagnostics=options["diagnostics"],
            diagnostic_interval=options["diagnostic_interval"],
            checkpoint_interval=options["checkpoint_interval"],
        ),
        output_directory=output,
        capture=PeriodicCapture(hydrodynamics=options["save_hydrodynamics"]),
        metadata=metadata,
    )
    return workflow.run(state, start_step=start, progress=progress)


def run_v04_translation(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    public_compilation_metadata: dict[str, object] | None = None,
):
    """Execute one periodic run inside a dynamically scoped TF32 policy."""

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
    spec = run_spec.simulation
    real_dtype = getattr(torch, spec.numerics.precision.value)
    requested = spec.execution.options["tf32"]
    if run_spec.dry_run:
        return _run_v04_translation_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            tf32_metadata=configure_tf32_execution(
                requested,
                real_dtype=real_dtype,
                device=spec.execution.options["device"],
            ),
            public_compilation_metadata=public_compilation_metadata,
        )
    device = resolve_execution_device(spec.execution.options["device"])
    with scoped_tf32_execution(
        requested,
        real_dtype=real_dtype,
        device=device,
    ) as tf32_metadata:
        return _run_v04_translation_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            tf32_metadata=tf32_metadata,
            public_compilation_metadata=public_compilation_metadata,
        )


def _reject_plane_restart(
    message: str,
    *,
    reason: str,
    step: int = 0,
    details: dict[str, object] | None = None,
    exception: Exception | None = None,
) -> None:
    error = SimulationRestartError(
        message,
        application_family="plane_complete_stress_beris_edwards",
        stage="restart_validation",
        step=step,
        details={"reason": reason, **(details or {})},
    )
    if exception is None:
        raise error
    raise error from exception


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Physical boundary conditions are declared once as immutable core contracts;
# legacy strings exist only at the current runtime adapter edge.
_LEGACY_BOUNDARIES = PLANE_FREE_SLIP_BOUNDARIES.to_legacy()
Q_BC = _LEGACY_BOUNDARIES["q"]
U_TANGENTIAL_BC = _LEGACY_BOUNDARIES["tangential_velocity"]
U_NORMAL_BC = _LEGACY_BOUNDARIES["normal_velocity"]
DISTORTION_ODD_Z_BC = _LEGACY_BOUNDARIES["distortion_odd_z"]
# div(u) lives in the DCT space, so pressure uses the same DCT modal multiplier
# space rather than an independently prescribed physical wall value.
PRESSURE_MODAL_BC = _LEGACY_BOUNDARIES["pressure_modal"]
# Tangential plug-flow handling. The recommended pilot default uses the
# fric=0 Stokes pseudoinverse and explicitly fixes <ux>=<uy>=0. This is a
# reference-frame/modeling choice, not a pressure gauge: free Q anchoring does
# not guarantee zero volume-averaged total nematic tangential force. Its active
# part alone can already be nonzero, e.g.
#
#   <f_x> is proportional to [<Q_xz>_{xy}]_{z=0}^{z=L_z}.
#
# A strict reproduction of the paper's inertial mean-momentum dynamics would
# require evolving the two tangential mean modes separately. Positive friction
# is retained as an optional sensitivity model but is not the default.
# Local source files whose contents determine the discrete Q/Stokes dynamics.
# Keep repository-relative names in metadata so runs made in different checkout
# locations remain directly comparable.
IMPLEMENTATION_SOURCE_FILES = PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES


def file_sha256(path):
    """Hash one implementation file for run-to-run provenance checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensors):
    """Hash an ordered collection of contiguous CPU tensor byte strings."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _restore_plane_progress(state, plan_interval, completed_steps,
                            spectral_refresh_interval, step_count, refresh_count):
    """Restore the v0.4 refresh clock, retaining its defensive error text."""
    if spectral_refresh_interval != plan_interval:
        raise RuntimeError("restored spectral-refresh counters are inconsistent")
    expected = _refresh_counters_from_completed_steps(completed_steps, plan_interval)
    if expected != (step_count, refresh_count):
        raise RuntimeError("restored spectral-refresh counters are inconsistent")
    state.completed_steps = completed_steps
    state.refresh_phase = expected


def restore_plane_state(engine, state, checkpoint, *, run_spec, diagnostics=None):
    """Validate every legacy tensor before copying it into the engine state."""
    if not isinstance(engine, Engine):
        raise TypeError("engine must be an Engine")
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    fields = engine.field_view(state)
    validate_plane_restore(
        checkpoint, run_spec,
        target_spatial={name: fields[name] for name in Q_COMPONENTS},
        target_spectral={name: fields[f"{name}.hat"] for name in Q_COMPONENTS},
        backend_restart=plane_backend_restart(run_spec),
    )
    targets: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint shape for {name}{suffix} differs")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
            targets.append((target, source, f"{name}{suffix}"))
    pending = tuple(
        (target, source.to(device=target.device))
        for target, source, _description in targets
    )
    for target, source in pending:
        target.copy_(source)
    state.constrained_current = False
    engine.prepare_constraints_(state, diagnostics=diagnostics)
    _restore_plane_progress(
        state, engine.plan.integrator.refresh_interval,
        checkpoint.completed_steps, checkpoint.spectral_refresh_interval,
        checkpoint.integrator_step_count, checkpoint.integrator_refresh_count,
    )
    if state.completed_steps != checkpoint.completed_steps:
        raise RuntimeError("restored completed-step count is inconsistent")
    return checkpoint.completed_steps


def _run_v04_plane_translation_owned(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    application_simulation: SimulationSpec | None = None,
    tf32_metadata: dict[str, object],
    components: PlaneBerisEdwardsRunComponents,
    public_compilation_metadata: dict[str, object] | None = None,
) -> RunResult | None:
    """Execute one validated Plane run through the supported application API.

    A dry-run returns None after resolving and optionally printing metadata.
    A simulation returns its RunResult. The caller owns progress
    presentation; omitting progress selects the quiet consecutive range.
    """

    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PlaneBerisEdwardsRunSpec")
    baseline_simulation = compose_plane_beris_edwards_simulation(components)
    if application_simulation is None:
        application_simulation = baseline_simulation
    elif not isinstance(application_simulation, SimulationSpec):
        raise TypeError("application_simulation must be SimulationSpec or None")
    lowering_plan = lower_simulation_spec(application_simulation)
    lifting_plan = lowering_plan.lifting_plan
    if lifting_plan is not None:
        raise ValueError("private Plane lifting construction is retired")
    q_assignment = application_simulation.boundaries.for_component("Qxx")
    q_physical_boundary_conditions = []
    for axis in range(application_simulation.geometry.domain.ndim):
        axis_kinds = {
            face.condition.kind.value
            for face in q_assignment.faces
            if face.axis == axis
        }
        if len(axis_kinds) != 1:
            raise ValueError(
                "Plane Q metadata requires matching boundary kinds on each "
                "axis pair"
            )
        q_physical_boundary_conditions.append(next(iter(axis_kinds)))
    transform_boundary_kind = {
        TransformKind.FFT: "periodic",
        TransformKind.DCT: "neumann",
        TransformKind.DST: "dirichlet",
    }
    q_evolved_boundary_conditions = tuple(
        transform_boundary_kind[kind]
        for kind in lowering_plan.basis_for("Qxx").transform_kinds
    )
    q_physical_boundary_conditions = tuple(q_physical_boundary_conditions)
    q_wall_model_note = (
        "prescribed static Dirichlet Q reconstructed from a homogeneous "
        "Dirichlet/DST remainder plus an affine lift"
        if lifting_plan is not None
        else "free/Neumann; matches the Fig. 4 free-anchoring branch"
    )
    domain = components.geometry.domain
    numerics = components.numerics
    physics = components.physics
    material = physics.material
    stokes = physics.stokes
    resolved_preset = components.preset
    time_stepping = components.time_stepping
    initial_condition = components.initial_condition
    execution = components.execution
    workflow_spec = components.workflow
    invocation = components.invocation
    is_restart = workflow_spec.restart_from is not None

    seed = initial_condition.seed
    dt = time_stepping.dt
    steps = workflow_spec.steps
    requested_device = execution.device
    device_normalization = execution_device_normalization(requested_device)
    device = device_normalization["resolved_execution_device"]
    real_dtype_name = numerics.precision.value
    real_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[real_dtype_name]
    spectral_dtype_name = "complex64" if real_dtype == torch.float32 else "complex128"
    resolved_torch_device = torch.device(device)
    cuda_device_name = None
    cuda_total_memory_bytes = None
    if resolved_torch_device.type == "cuda":
        cuda_properties = torch.cuda.get_device_properties(resolved_torch_device)
        cuda_device_name = cuda_properties.name
        cuda_total_memory_bytes = int(cuda_properties.total_memory)
    runtime_environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "resolved_device": str(resolved_torch_device),
        "device_type": resolved_torch_device.type,
        "cuda_device_name": cuda_device_name,
        "cuda_total_memory_bytes": cuda_total_memory_bytes,
    }
    pointwise_kernels = BerisEdwardsPointwiseKernels(
        execution.pointwise_execution
    )
    pointwise_execution_metadata = pointwise_kernels.metadata()

    Nx, Ny, Nz = domain.shape
    Lx, Ly, Lz = domain.lengths
    save_interval = workflow_spec.save_interval
    save_start_step = workflow_spec.save_start_step
    diagnostic_interval = workflow_spec.diagnostic_interval
    diagnostics_enabled = workflow_spec.diagnostics
    save_hydrodynamics = workflow_spec.save_hydrodynamics
    alignment_parameter = material.flow_alignment
    zero_mode_policy = stokes.tangential_zero_mode_policy.value
    dealias_rule = numerics.dealias_rule.value
    projected_transform_execution = (
        numerics.projected_transform_execution.value
    )
    spectral_storage = numerics.spectral_storage.value
    transform_execution_order = numerics.transform_execution_order.value
    projected_transform_execution_metadata = {
        "requested": projected_transform_execution,
        "effective": projected_transform_execution,
        "fallback_allowed": False,
        "fallback_reason": None,
        "truncated_real_basis_axes": (
            projected_transform_execution == "truncated"
        ),
        "spectral_storage": spectral_storage,
        "backend_storage_shape_preserved": True,
        "full_spectral_storage_preserved": (
            spectral_storage == "full_complex"
        ),
    }
    spectral_shape = [
        Nx,
        Ny // 2 + 1 if spectral_storage == "hermitian_half" else Ny,
        Nz,
    ]

    # Write Q = q (nn-I/3), where q = 3 S/2 in the declared convention.
    # Equating (L1/2) |grad Q|^2 with (K/2) |grad n|^2 gives
    # K = 2 L1 q_eq^2 and therefore L1 = K/(2 q_eq^2).  At the Shendruk
    # bulk equilibrium S_eq=1/3, q_eq=1/2 and L1=2K.  Using q_eq explicitly
    # avoids the scalar-amplitude ambiguity in the paper's printed mapping.
    zeta = resolved_preset.zeta
    frank_k = resolved_preset.frank_k
    S_bulk = resolved_preset.equilibrium_s
    q_equilibrium_amplitude = resolved_preset.equilibrium_q_amplitude
    ldg_l1 = resolved_preset.ldg_l1
    rotational_viscosity = resolved_preset.rotational_viscosity
    ldg_a_over_gamma = resolved_preset.ldg_a_over_gamma
    ldg_b_over_gamma = resolved_preset.ldg_b_over_gamma
    ldg_c_over_gamma = resolved_preset.ldg_c_over_gamma
    ldg_l1_over_gamma = resolved_preset.ldg_l1_over_gamma

    # The active stress in this code is beta * alpha * Q.  beta=-1 therefore
    # matches the paper's -zeta Q convention when alpha=zeta.
    alpha_value = zeta
    beta = material.beta
    fric = stokes.friction
    eta = stokes.viscosity

    implementation_provenance = {
        "schema_version": 1,
        "files": {
            relative_path: file_sha256(
                _PROJECT_ROOT / relative_path
            )
            for relative_path in IMPLEMENTATION_SOURCE_FILES
        },
    }

    metadata = {
        "schema_version": 2,
        "application": "plane_complete_stress_beris_edwards",
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
        "normalization": device_normalization,
        "validation_config_sha256": invocation.validation_config_sha256,
        "implementation_provenance": implementation_provenance,
        "runtime_environment": runtime_environment,
        "solver": {
            "shape": [Nx, Ny, Nz],
            "spectral_shape": spectral_shape,
            "lengths": [Lx, Ly, Lz],
            "dt": dt,
            "steps": steps,
            "save_interval": save_interval,
            "real_dtype": real_dtype_name,
            "spectral_dtype": spectral_dtype_name,
            "transform_execution_order": transform_execution_order,
            "spectral_storage": spectral_storage,
        },
        "model": {
            "name": "active_nematics",
            "variant": "beris_edwards_complete_nematic_stress_stokes",
            "stage": "reusable_beris_edwards_complete_stress_stokes",
            "Q_convention": Q_convention_metadata(),
            "q_dynamics": {
                "name": "Beris-Edwards (Shendruk parameterization)",
                "implementation": (
                    "pssolver.models.active_nematics.BerisEdwardsQNonlinearModel"
                ),
                "equation": "(partial_t+u.grad)Q-S(E,Omega,Q)=H/gamma",
                "flow_alignment_form": "full_beris_edwards",
                "molecular_field": "one_constant_landau_de_gennes",
                "pointwise_execution": execution.pointwise_execution,
                "raw_coefficients": {
                    "A": material.ldg_a,
                    "B": material.ldg_b,
                    "C": material.ldg_c,
                    "L1": ldg_l1,
                    "rotational_viscosity_gamma": rotational_viscosity,
                    "flow_alignment_lambda": alignment_parameter,
                },
                "evolution_coefficients": {
                    "A_over_gamma": ldg_a_over_gamma,
                    "B_over_gamma": ldg_b_over_gamma,
                    "C_over_gamma": ldg_c_over_gamma,
                    "L1_over_gamma": ldg_l1_over_gamma,
                },
                "frank_to_ldg_mapping": {
                    "frank_K": frank_k,
                    "tensor_amplitude_q_eq": q_equilibrium_amplitude,
                    "formula": "L1=K/(2*q_eq^2), q_eq=3*S_eq/2",
                    "paper_notation_note": (
                        "The printed mapping uses ambiguous scalar-amplitude "
                        "notation; this script uses the declared Q convention."
                    ),
                },
            },
            "flow_dynamics": {
                "momentum_equation": (
                    "0=-grad(p)+eta*lap(u)-fric*u+div(Pi_nematic)"
                ),
                "regime": "quasistatic_incompressible_stokes_brinkman",
                "nematic_stress": (
                    "complete_one_constant_beris_edwards_up_to_isotropic_pressure"
                ),
                "stress_components": [
                    "reactive",
                    "one_constant_distortion",
                    "active",
                ],
                "reactive_stress": (
                    "2*lambda*M*(Q:H)-lambda*(H*M+M*H)+Q*H-H*Q; "
                    "M=Q+I/3"
                ),
                "distortion_stress": (
                    "-L1*(partial_i Q_kl)*(partial_j Q_kl)"
                ),
                "active_stress": "beta*alpha*Q; beta=-1 gives -zeta*Q",
                "molecular_field_in_stress": "raw_H_not_H_over_gamma",
                "molecular_field_linear_space": (
                    execution.molecular_field_linear_space
                ),
                "stress_divergence_sum_space": (
                    execution.stress_divergence_sum_space
                ),
                "pointwise_execution": execution.pointwise_execution,
                "isotropic_stress": "absorbed_into_incompressible_pressure",
                "viscous_stress": "handled_by_eta_laplacian_in_stokes_operator",
            },
            "parameters": {
                "activity_number": physics.shendruk_request.activity_number,
                "zeta": zeta,
                "frank_K": frank_k,
                "ldg_A": material.ldg_a,
                "ldg_B": material.ldg_b,
                "ldg_C": material.ldg_c,
                "ldg_L1": ldg_l1,
                "rotational_viscosity_gamma": rotational_viscosity,
                "flow_alignment_lambda": alignment_parameter,
                "alpha": alpha_value,
                "beta": beta,
                "S_initial": initial_condition.initial_s,
                "S_bulk": S_bulk,
                "fric": fric,
                "eta": eta,
            },
        },
        "boundary_conditions": {
            "Q": q_physical_boundary_conditions,
            "Q_assignment": [
                application_simulation.boundaries.for_component(
                    component
                ).to_metadata()
                for component in (
                    lifting_plan.component_order
                    if lifting_plan is not None
                    else ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
                )
            ],
            "Q_evolved_representation": (
                "homogeneous_remainder"
                if lifting_plan is not None
                else "physical_field"
            ),
            "Q_evolved_boundary_conditions": q_evolved_boundary_conditions,
            "velocity_tangential": U_TANGENTIAL_BC,
            "velocity_normal": U_NORMAL_BC,
            "pressure": PRESSURE_MODAL_BC,
            "velocity_wall_model_note": (
                "homogeneous kinematic free-slip; not zero total nematic traction"
            ),
        },
        "numerics": {
            "dealiasing": {
                "rule": dealias_rule,
                "fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
                "projected_transform_execution": (
                    projected_transform_execution_metadata
                ),
                "nematic_force_evaluation": (
                    "project_H_then_algebraic_and_distortion_stresses_then_force"
                ),
                "monolithic_quintic_galerkin": False,
                "distortion_stress_z_parity": {
                    "even_components": ["xx", "xy", "yx", "yy", "zz"],
                    "odd_components": ["xz", "yz", "zx", "zy"],
                    "even_basis": Q_BC,
                    "odd_basis": DISTORTION_ODD_Z_BC,
                },
            },
            "velocity_zero_mode": zero_mode_policy,
            "zero_mode_force": "total_nematic_tangential_force",
            "pressure_solver": "free_slip_modal_schur_complement",
            "pressure_residual_diagnostics": diagnostics_enabled,
            "molecular_field_linear_space": (
                execution.molecular_field_linear_space
            ),
            "stress_divergence_sum_space": (
                execution.stress_divergence_sum_space
            ),
            "pointwise_kernels": pointwise_execution_metadata,
            "q_gradient_reuse": {
                "enabled": not execution.disable_q_gradient_reuse,
                "scope": "single_static_to_nonlinear_evaluation",
                "mutation_guard": "spatial_and_spectral_tensor_versions",
            },
            "transforms": {
                "execution_order": transform_execution_order,
                "spectral_storage": spectral_storage,
                "physical_shape": [Nx, Ny, Nz],
                "spectral_shape": spectral_shape,
                "hermitian_axis": (
                    numerics.hermitian_axis
                    if spectral_storage == "hermitian_half"
                    else None
                ),
                "basis_and_normalization_changed": False,
                "projected_transform_execution": (
                    projected_transform_execution
                ),
            },
            "precision": {
                "real_dtype": real_dtype_name,
                "spectral_dtype": spectral_dtype_name,
                **tf32_metadata,
            },
            "spectral_refresh": {
                "mode": time_stepping.spectral_refresh.mode,
                "requested_interval_time": (
                    time_stepping.spectral_refresh.requested_interval_time
                ),
                "requested_interval_steps": (
                    time_stepping.spectral_refresh.requested_interval_steps
                ),
                "effective_interval_steps": (
                    time_stepping.spectral_refresh.effective_interval_steps
                ),
                "effective_interval_time": (
                    time_stepping.spectral_refresh.effective_interval_time
                ),
                "phase_origin_step": 0,
            },
        },
        "benchmark_target": "Shendruk et al. PRE 98, 010601(R) (2018), Fig. 4",
        "reproduction_status": "development_reusable_beris_edwards_stokes_stage",
        "scan_variable": "activity_number",
        "parameterization": physics.shendruk_request.parameterization,
        "activity_number": physics.shendruk_request.activity_number,
        "activity_number_definition": "H*sqrt(zeta/K)",
        "zeta": zeta,
        "frank_k": frank_k,
        "ldg_l1": ldg_l1,
        "q_equilibrium_amplitude": q_equilibrium_amplitude,
        "q_elastic_relaxation": ldg_l1_over_gamma,
        "height": Lz,
        "domain": [Lx, Ly, Lz],
        "shape": [Nx, Ny, Nz],
        "dt": dt,
        "steps": steps,
        "save_start_step": save_start_step,
        "save_interval": save_interval,
        "diagnostic_interval": diagnostic_interval,
        "seed": seed,
        "device": device,
        "dtype": real_dtype_name,
        "transform_execution_order": transform_execution_order,
        "spectral_storage": spectral_storage,
        "molecular_field_linear_space": (
            execution.molecular_field_linear_space
        ),
        "stress_divergence_sum_space": (
            execution.stress_divergence_sum_space
        ),
        "pointwise_execution": execution.pointwise_execution,
        "tf32": execution.tf32,
        "q_boundary_conditions": q_physical_boundary_conditions,
        "q_evolved_boundary_conditions": q_evolved_boundary_conditions,
        "tangential_velocity_boundary_conditions": U_TANGENTIAL_BC,
        "normal_velocity_boundary_conditions": U_NORMAL_BC,
        "velocity_wall_model": "free-slip",
        "q_wall_model_note": q_wall_model_note,
        "zero_mode_policy": zero_mode_policy,
        "dealias_rule": dealias_rule,
        "dealias_fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
        "projected_transform_execution": projected_transform_execution,
        "ldg_coefficients": {
            "A": material.ldg_a,
            "B": material.ldg_b,
            "C": material.ldg_c,
        },
        "gamma": material.gamma,
        "flow_alignment": alignment_parameter,
        "eta": eta,
        "friction": fric,
        "active_stress_beta": beta,
        "S_initial": initial_condition.initial_s,
        "S_bulk": S_bulk,
        "initial_condition": {
            "name": "extruded_analytic_periodic_defect_gas_2d",
            "source": "PSSolver constructed",
            "paper_identical": False,
            "seed": seed,
            "S_initial": initial_condition.initial_s,
            "twist_amplitude": initial_condition.twist_amplitude,
            "twist_modes": initial_condition.twist_modes,
        },
        "initial_defect_gas": {
            "num_pairs": initial_condition.num_defect_pairs,
            "minimum_separation": initial_condition.defect_min_separation,
            "core_radius": initial_condition.defect_core_radius,
            "S_initial": initial_condition.initial_s,
            "background_angle": initial_condition.background_angle,
        },
        "initial_neumann_twist": {
            "rms_amplitude_radians": initial_condition.twist_amplitude,
            "dct_modes": initial_condition.twist_modes,
        },
        "save_hydrodynamics": save_hydrodynamics,
        "model_limitations": [
            (
                "quasistatic zero-Reynolds-number Stokes rather than the paper's "
                "inertial momentum equation"
            ),
            (
                "homogeneous velocity free-slip is kinematic and does not impose "
                "zero total nematic traction"
            ),
            (
                "tangential plug-mode treatment is a modeling choice: zero_mean "
                "removes total forcing, while friction adds drag"
            ),
            (
                "initial condition is PSSolver-constructed and not verified "
                "paper-identical"
            ),
            (
                "staged spectral filtering differs from the paper's "
                "finite-difference/LB discretization"
            ),
        ],
    }
    if public_compilation_metadata is not None:
        metadata["public_compilation"] = dict(public_compilation_metadata)
    # Reject unsupported cross-runtime or numerically incompatible restart before
    # creating an output directory, generating Q, or constructing either solver.
    restart_kind = None
    engine_header = None
    if is_restart:
        from pssolver.io.checkpoint import read_checkpoint_json
        restart_metadata = read_checkpoint_json(
            Path(workflow_spec.restart_from) / "checkpoint.json",
            operation="plane_checkpoint_import",
        )
        if restart_metadata.get("format_kind") == "engine_state":
            engine_header = engine_state.read_header(
                workflow_spec.restart_from,
                plan_identity_sha256=translate_plane(run_spec).identity_sha256,
                dtype=real_dtype, batch=1,
            )
            restart_kind = "engine_state"
            if save_start_step > engine_header["completed_steps"] + steps:
                _reject_plane_restart(
                    "save_start_step must not exceed the absolute final step",
                    reason="save_start_step_after_final_step",
                    step=engine_header["completed_steps"],
                    details={
                        "save_start_step": save_start_step,
                        "absolute_final_step": engine_header["completed_steps"] + steps,
                    },
                )
    if is_restart and restart_kind != "engine_state":
        try:
            restart_header = read_plane_checkpoint_header(
                workflow_spec.restart_from
            )
        except SimulationRestartError:
            raise
        except CheckpointError:
            raise
        except Exception as exc:
            _reject_plane_restart(
                str(exc) or "Plane checkpoint preflight failed",
                reason="checkpoint_preflight_failed",
                details={"legacy_exception_type": type(exc).__name__},
                exception=exc,
            )
        restart_step = getattr(restart_header, "completed_steps", 0)
        if restart_header.runtime_path is not execution.runtime_path:
            _reject_plane_restart(
                "cross-runtime Plane checkpoint restart is unsupported",
                reason="runtime_path_mismatch",
                step=restart_step,
                details={
                    "checkpoint_runtime_path": restart_header.runtime_path.value,
                    "target_runtime_path": execution.runtime_path.value,
                },
            )
        if restart_header.format_version == (
            PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
        ):
            _reject_plane_restart(
                "legacy Plane checkpoint requires explicit identity upgrade",
                reason="legacy_identity_upgrade_required",
                step=restart_step,
            )
        checkpoint_identity = restart_header.compatibility_identity
        if checkpoint_identity is None:
            _reject_plane_restart(
                "checkpoint compatibility identity is missing",
                reason="compatibility_identity_missing",
                step=restart_step,
            )
        checkpoint_forward = (
            checkpoint_identity.forward_dynamics.to_metadata()["payload"]
        )
        checkpoint_run_spec = checkpoint_forward.get("run_spec_dynamics")
        target_run_spec = plane_run_spec_dynamics_identity(
            run_spec
        ).to_metadata()
        if checkpoint_run_spec != target_run_spec:
            _reject_plane_restart(
                "checkpoint forward dynamics does not match target",
                reason="forward_dynamics_mismatch",
                step=restart_step,
            )
        checkpoint_lifting = checkpoint_forward.get("lifting_dynamics")
        if lifting_plan is None:
            target_lifting_plan_sha256 = None
        else:
            target_lifting_plan_sha256 = lifting_plan.canonical_sha256()
        if checkpoint_lifting is None:
            checkpoint_lifting_plan_sha256 = None
        elif isinstance(checkpoint_lifting, dict):
            lifting_identity = checkpoint_lifting.get("lifting")
            checkpoint_lifting_plan_sha256 = (
                lifting_identity.get("plan_sha256")
                if isinstance(lifting_identity, dict)
                else None
            )
        else:
            checkpoint_lifting_plan_sha256 = None
        if checkpoint_lifting_plan_sha256 != target_lifting_plan_sha256:
            _reject_plane_restart(
                "checkpoint static lifting dynamics does not match target",
                reason="static_lifting_dynamics_mismatch",
                step=restart_step,
            )
        if save_start_step > restart_header.completed_steps + steps:
            _reject_plane_restart(
                "save_start_step must not exceed the absolute final step",
                reason="save_start_step_after_final_step",
                step=restart_step,
                details={
                    "save_start_step": save_start_step,
                    "absolute_final_step": restart_header.completed_steps + steps,
                },
            )

    metadata["initial_condition"]["runtime_construction_only"] = is_restart
    metadata["initial_condition"]["artifacts_written"] = not is_restart
    if is_restart:
        metadata["initial_condition"]["trajectory_state_source"] = (
            "authenticated_checkpoint"
        )
        metadata["initial_condition"]["checkpoint_path"] = str(
            workflow_spec.restart_from.resolve()
        )
    else:
        metadata["initial_condition"]["trajectory_state_source"] = (
            "generated_initial_condition"
        )
    if emit_metadata:
        print(json.dumps(metadata, indent=2))
    if invocation.dry_run:
        return None

    output_dir = workflow_spec.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix pilot outputs in nonempty {output_dir}")

    q_2d = create_initial_condition(
        "analytic_periodic_defect_gas_2d",
        shape=(Nx, Ny),
        lengths=(Lx, Ly),
        num_defect_pairs=initial_condition.num_defect_pairs,
        min_separation=initial_condition.defect_min_separation,
        core_radius=initial_condition.defect_core_radius,
        seed=seed,
        S_initial=initial_condition.initial_s,
        background_angle=initial_condition.background_angle,
        dtype=real_dtype,
    )
    if not is_restart:
        defect_positions, defect_charges = sample_periodic_neutral_defects_2d(
            lengths=(Lx, Ly),
            num_defect_pairs=initial_condition.num_defect_pairs,
            min_separation=initial_condition.defect_min_separation,
            seed=seed,
        )

    q_initial_condition = create_initial_condition(
        "extruded_2d_twist",
        shape=(Nx, Ny, Nz),
        Q_2d=q_2d,
        boundary_conditions=Q_BC,
        seed=seed,
        twist_amplitude=initial_condition.twist_amplitude,
        twist_modes=initial_condition.twist_modes,
        dtype=real_dtype,
    )
    Qxx_0 = q_initial_condition["Qxx"]
    Qxy_0 = q_initial_condition["Qxy"]
    Qxz_0 = q_initial_condition["Qxz"]
    Qyy_0 = q_initial_condition["Qyy"]
    Qyz_0 = q_initial_condition["Qyz"]
    metadata["initial_condition"]["raw_q_sha256"] = tensor_sha256(
        (Qxx_0, Qxy_0, Qxz_0, Qyy_0, Qyz_0)
    )
    initial_values = {
        "Qxx": Qxx_0,
        "Qxy": Qxy_0,
        "Qxz": Qxz_0,
        "Qyy": Qyy_0,
        "Qyz": Qyz_0,
    }
    plan = translate_plane(run_spec)
    engine = Engine(plan, device=device, dtype=real_dtype)
    state = engine.initial_state(
        {name: value.to(device=device) for name, value in initial_values.items()}
    )
    fields = engine.field_view(state)
    spectral_projector = engine.projector
    metadata["runtime_construction"] = {
        "plan_identity_sha256": engine.plan.identity_sha256,
        "engine": "pssolver.engine.engine.Engine",
        "operation_order": [
            "prepare_constraints", "explicit", "euler", "project", "inverse",
        ],
    }
    metadata["runtime_selection"] = {
        **run_spec.runtime_selection_metadata(),
        "requested": execution.runtime_path.value,
        "effective": execution.runtime_path.value,
        "adapter": "Engine",
        "fallback_used": False,
        "separated_architecture": None,
    }
    metadata["initial_condition"]["projected_q_sha256"] = tensor_sha256(
        tuple(
            fields[name]
            for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
        )
    )
    q_runtime_boundaries = fields["Qxx.bc"]
    metadata["retained_q_modes"] = (
        spectral_projector.retained_axis_counts(q_runtime_boundaries)
    )
    metadata["retained_normal_velocity_modes"] = (
        spectral_projector.retained_axis_counts(U_NORMAL_BC)
    )
    if not is_restart:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir / "Q2D_initial.npy",
            np.stack(
                [
                    q_2d[name].numpy()
                    for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
                ],
                axis=-1,
            ),
        )
        np.savetxt(
            output_dir / "Q2D_defects.csv",
            np.column_stack((defect_positions, defect_charges)),
            delimiter=",",
            header="x,y,charge",
            comments="",
        )
    start_step = 0
    restored_diagnostics = {}
    metadata["restart"] = {"kind": "fresh"}
    if is_restart:
        try:
            if restart_kind == "engine_state":
                state = engine_state.read(
                    workflow_spec.restart_from, plan_identity_sha256=plan.identity_sha256,
                    dtype=real_dtype, batch=1, device=device,
                )
                start_step = state.completed_steps
                # Diagnostics are caller-owned and not in the checkpoint carrier.
                state.constrained_current = False
                engine.prepare_constraints_(state, diagnostics=restored_diagnostics)
                metadata["restart"] = {
                    "kind": "engine_state",
                    "source": str(Path(workflow_spec.restart_from).expanduser().resolve()),
                    "source_runtime_path": engine_header["run_provenance"]["legacy_configuration"]["runtime_path"],
                    "source_completed_steps": start_step,
                }
            else:
                checkpoint = load_plane_checkpoint(workflow_spec.restart_from)
                start_step = restore_plane_state(
                    engine, state, checkpoint, run_spec=run_spec,
                    diagnostics=restored_diagnostics,
                )
                metadata["restart"] = {
                    "kind": "same_backend_plane_workflow_checkpoint",
                    "source": str(Path(workflow_spec.restart_from).expanduser().resolve()),
                    "source_runtime_path": checkpoint.runtime_path.value,
                    "source_runtime_identity_sha256": checkpoint.runtime_identity_sha256,
                    "source_completed_steps": checkpoint.completed_steps,
                    "cross_backend_adapter_used": False,
                }
        except (SimulationRestartError, CheckpointError):
            raise
        except Exception as exc:
            raise SimulationRestartError(
                str(exc) or "Plane checkpoint restore failed",
                application_family="plane_complete_stress_beris_edwards",
                stage="restart_restore", step=state.completed_steps,
                details={
                    "reason": "checkpoint_restore_failed",
                    "legacy_exception_type": type(exc).__name__,
                },
            ) from exc
        metadata["initial_condition"]["used_for_runtime_construction_only"] = True
    metadata["workflow"] = {
        "schema_version": 1,
        "runtime_path": execution.runtime_path.value,
        "checkpoint_format_version": "engine_state.v1",
        "cross_backend_restart_supported": True,
        "start_step": start_step,
        "requested_additional_steps": workflow_spec.steps,
        "completion_marker_order": "metadata_then_COMPLETE",
    }
    workflow = Runner(
        engine,
        schedule=RunSchedule(
            steps=workflow_spec.steps, save_interval=workflow_spec.save_interval,
            save_start_step=workflow_spec.save_start_step, diagnostics=workflow_spec.diagnostics,
            diagnostic_interval=workflow_spec.diagnostic_interval,
            checkpoint_interval=workflow_spec.checkpoint_interval,
        ),
        output_directory=output_dir,
        capture=PlaneCapture(
            hydrodynamics=workflow_spec.save_hydrodynamics,
            runtime_path=execution.runtime_path.value, viscosity=stokes.viscosity,
            friction=stokes.friction,
        ),
        metadata=metadata,
    )
    workflow.constraint_diagnostics.update(restored_diagnostics)
    return workflow.run(state, start_step=start_step, progress=progress)



def run_v04_plane_translation(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    application_simulation: SimulationSpec | None = None,
    public_compilation_metadata: dict[str, object] | None = None,
) -> RunResult | None:
    """Execute one Plane run inside a dynamically scoped TF32 policy."""

    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PlaneBerisEdwardsRunSpec")
    components = decompose_plane_beris_edwards_run_spec(run_spec)
    execution = components.execution
    invocation = components.invocation
    real_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[components.numerics.precision.value]
    requested_device = execution.device
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    if invocation.dry_run:
        return _run_v04_plane_translation_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            application_simulation=application_simulation,
            tf32_metadata=configure_tf32_execution(
                execution.tf32,
                real_dtype=real_dtype,
                device=device,
            ),
            components=components,
            public_compilation_metadata=public_compilation_metadata,
        )
    with scoped_tf32_execution(
        execution.tf32,
        real_dtype=real_dtype,
        device=device,
    ) as tf32_metadata:
        return _run_v04_plane_translation_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            application_simulation=application_simulation,
            tf32_metadata=tf32_metadata,
            components=components,
            public_compilation_metadata=public_compilation_metadata,
        )
