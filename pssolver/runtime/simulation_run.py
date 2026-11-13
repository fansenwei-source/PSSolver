"""Prepare one assembled simulation and hand its state to the shared Runner."""

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pssolver.core.numerics import DEALIAS_RULE_FRACTIONS
from pssolver.engine.execution_device import (
    configure_tf32_execution,
    execution_device_normalization,
    resolve_execution_device,
    scoped_tf32_execution,
)
from pssolver.io import engine_state
from pssolver.io.checkpoint import CheckpointError, read_checkpoint_json
from pssolver.io.initial_snapshot import load_q_snapshot
from pssolver.models.active_nematics.beris_edwards import (
    POINTWISE_COMPILE_BACKEND,
    POINTWISE_COMPILE_MODE,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.models.active_nematics.initial_conditions import (
    create_initial_condition,
    sample_periodic_neutral_defects_2d,
)
from pssolver.models.active_nematics.q_tensor import Q_convention_metadata
from pssolver.runtime.channel_observations import ChannelCapture
from pssolver.runtime.observations import PeriodicCapture
from pssolver.runtime.plane_observations import PlaneCapture
from pssolver.runtime.runner import Runner, RunSchedule


_APPLICATIONS = {
    "periodic_box": "periodic_complete_stress_beris_edwards",
    "plane_slab": "plane_complete_stress_beris_edwards",
    "rectangular_channel": "channel_complete_stress_beris_edwards",
}
_CHECKPOINT_OPERATIONS = {
    "periodic_box": "periodic_checkpoint_import",
    "plane_slab": "plane_checkpoint_import",
    "rectangular_channel": "complete_stress_channel_checkpoint_import",
}
_Q_BOUNDARY = ("periodic", "periodic", "neumann")


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    """State and output scheduler after construction and any restore."""

    engine: object
    state: object
    runner: Runner
    start_step: int
    construction_diagnostics: dict[str, object]


_PLANE_IMPLEMENTATION_FILES = (
    "pssolver/backends/__init__.py",
    "pssolver/backends/bounded.py",
    "pssolver/backends/tensor_product.py",
    "pssolver/operators/__init__.py",
    "pssolver/operators/projection.py",
    "pssolver/operators/tensor_divergence.py",
    "pssolver/planning/__init__.py",
    "pssolver/linear_solvers/__init__.py",
    "pssolver/linear_solvers/stokes/__init__.py",
    "pssolver/linear_solvers/stokes/plane_free_slip.py",
    "pssolver/__init__.py",
    "pssolver/runtime/__init__.py",
    "pssolver/core/__init__.py",
    "pssolver/core/boundary.py",
    "pssolver/core/domain.py",
    "pssolver/core/geometry.py",
    "pssolver/core/numerics.py",
    "pssolver/geometries/__init__.py",
    "pssolver/geometries/tensor_product.py",
    "pssolver/geometries/plane_numerics.py",
    "pssolver/models/active_nematics/__init__.py",
    "pssolver/models/active_nematics/fields.py",
    "pssolver/models/active_nematics/q_tensor.py",
    "pssolver/models/active_nematics/beris_edwards.py",
    "pssolver/models/active_nematics/stokes.py",
    "pssolver/models/active_nematics/initial_conditions.py",
    "pssolver/presets/__init__.py",
    "pssolver/presets/shendruk.py",
    "pssolver/linear_solvers/stokes/plane_free_slip_stateless.py",
    "pssolver/cli/__init__.py",
    "pssolver/cli/plane.py",
    "pssolver/engine/engine.py",
    "pssolver/engine/program.py",
    "pssolver/engine/constraints.py",
    "pssolver/engine/groups.py",
    "pssolver/engine/terms.py",
    "pssolver/planning/assemble.py",
    "pssolver/planning/capabilities.py",
    "pssolver/planning/engine_plan.py",
    "pssolver/presets/beris_edwards.py",
    "pssolver/models/active_nematics/engine_terms.py",
    "pssolver/runtime/runner.py",
    "pssolver/runtime/plane_observations.py",
    "pssolver/api/runner.py",
    "pssolver/experimental/boundary_pilots.py",
    "pssolver/api/simulation.py",
    "pssolver/api/declarations.py",
    "pssolver/api/models.py",
    "pssolver/api/lowering.py",
    "pssolver/runtime/simulation_run.py",
    "pssolver/engine/execution_device.py",
)


def _pointwise_metadata(execution):
    try:
        triton = distribution_version("triton")
    except PackageNotFoundError:
        triton = None
    enabled = execution == "compile"
    return {
        "requested": execution,
        "effective": execution,
        "fallback_allowed": False,
        "fallback_reason": None,
        "compile": {
            "enabled": enabled,
            "backend": POINTWISE_COMPILE_BACKEND if enabled else None,
            "mode": POINTWISE_COMPILE_MODE if enabled else None,
            "dynamic": False if enabled else None,
            "fullgraph": True if enabled else None,
            "torch_version": str(torch.__version__),
            "triton_version": triton,
        },
    }


def _q_assignment(component, bases):
    return {
        "component": component,
        "semantic": "physical",
        "faces": [
            {"axis": axis, "side": side, "condition": {"kind": basis, "is_homogeneous": True}}
            for axis, basis in enumerate(bases)
            for side in ("lower", "upper")
        ],
    }


def _plane_metadata(compiled, *, normalization, tf32_metadata):
    """Keep the published Plane metadata paths using the lowered Plan and preset."""
    source = compiled.source
    plan = compiled.plan
    construction = compiled._run.construction
    preset, refresh = construction.plane_preset, construction.refresh
    numerics = source.numerics
    initial = SimpleNamespace(**source.initial_condition.parameters)
    workflow = source.output
    parameters = plan.constraints[0].arguments
    execution = SimpleNamespace(
        molecular_field_linear_space=parameters["molecular_field_linear_space"],
        stress_divergence_sum_space=parameters["stress_divergence_sum_space"],
        pointwise_execution=plan.explicit[0].pointwise_execution,
        disable_q_gradient_reuse=not plan.explicit[0].q_gradient_reuse,
    )
    device_normalization = normalization
    device = normalization["resolved_execution_device"]
    resolved_device = torch.device(device)
    cuda_properties = (
        torch.cuda.get_device_properties(resolved_device)
        if resolved_device.type == "cuda"
        else None
    )
    runtime_environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "resolved_device": str(resolved_device),
        "device_type": resolved_device.type,
        "cuda_device_name": None if cuda_properties is None else cuda_properties.name,
        "cuda_total_memory_bytes": None
        if cuda_properties is None
        else int(cuda_properties.total_memory),
    }
    project_root = Path(__file__).resolve().parents[2]
    implementation_provenance = {
        "schema_version": 1,
        "files": {
            path: hashlib.sha256((project_root / path).read_bytes()).hexdigest()
            for path in _PLANE_IMPLEMENTATION_FILES
        },
    }
    nx, ny, nz = plan.geometry.shape
    lx, ly, lz = plan.geometry.lengths
    nxh, nyh, nzh = nx, ny, nz
    if plan.geometry.spectral_storage == "hermitian_half":
        nyh = ny // 2 + 1
    # Keep the old metadata spellings; the values are read from the one Plan.
    Nx, Ny, Nz = nx, ny, nz
    Lx, Ly, Lz = lx, ly, lz
    spectral_shape = [nxh, nyh, nzh]
    q_physical_boundary_conditions = plan.fields[0].boundary_conditions
    q_evolved_boundary_conditions = q_physical_boundary_conditions
    u_tangential_bc, _, u_normal_bc = plan.fields[1].boundary_conditions
    pressure_modal_bc = plan.fields[2].boundary_conditions
    distortion_odd_z_bc = parameters["distortion_odd_boundary_conditions"]
    dealias_rule = plan.dealias.rule
    projected_transform_execution = plan.dealias.transform_execution
    spectral_storage = plan.geometry.spectral_storage
    projected_transform_execution_metadata = {
        "requested": projected_transform_execution,
        "effective": projected_transform_execution,
        "fallback_allowed": False,
        "fallback_reason": None,
        "truncated_real_basis_axes": projected_transform_execution == "truncated",
        "spectral_storage": spectral_storage,
        "backend_storage_shape_preserved": True,
        "full_spectral_storage_preserved": spectral_storage == "full_complex",
    }
    real_dtype_name = numerics.dtype
    spectral_dtype_name = "complex64" if real_dtype_name == "float32" else "complex128"
    transform_execution_order = plan.geometry.transform_execution_order
    pointwise_execution_metadata = _pointwise_metadata(execution.pointwise_execution)
    q_wall_model_note = "free/Neumann; matches the Fig. 4 free-anchoring branch"
    zeta, frank_k, ldg_l1 = preset.zeta, preset.frank_k, preset.ldg_l1
    S_bulk = preset.equilibrium_s
    q_equilibrium_amplitude = preset.equilibrium_q_amplitude
    rotational_viscosity = preset.rotational_viscosity
    ldg_a_over_gamma = preset.ldg_a_over_gamma
    ldg_b_over_gamma = preset.ldg_b_over_gamma
    ldg_c_over_gamma = preset.ldg_c_over_gamma
    ldg_l1_over_gamma = preset.ldg_l1_over_gamma
    alpha_value, beta = zeta, parameters["beta_value"]
    fric, eta = parameters["friction"], parameters["viscosity"]
    zero_mode_policy = parameters["zero_mode_policy"]
    alignment_parameter = source.model.flow_alignment
    dt, steps, seed = plan.integrator.dt, workflow.steps, initial.seed
    save_interval, save_start_step = workflow.save_interval, workflow.save_start_step
    diagnostic_interval = workflow.diagnostic_interval
    diagnostics_enabled, save_hydrodynamics = workflow.diagnostics, workflow.save_hydrodynamics

    return {
        "normalization": device_normalization,
        "validation_config_sha256": source.output.validation_config_sha256,
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
                "implementation": ("pssolver.models.active_nematics.BerisEdwardsQNonlinearModel"),
                "equation": "(partial_t+u.grad)Q-S(E,Omega,Q)=H/gamma",
                "flow_alignment_form": "full_beris_edwards",
                "molecular_field": "one_constant_landau_de_gennes",
                "pointwise_execution": execution.pointwise_execution,
                "raw_coefficients": {
                    "A": source.model.ldg_a,
                    "B": source.model.ldg_b,
                    "C": source.model.ldg_c,
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
                "momentum_equation": ("0=-grad(p)+eta*lap(u)-fric*u+div(Pi_nematic)"),
                "regime": "quasistatic_incompressible_stokes_brinkman",
                "nematic_stress": ("complete_one_constant_beris_edwards_up_to_isotropic_pressure"),
                "stress_components": [
                    "reactive",
                    "one_constant_distortion",
                    "active",
                ],
                "reactive_stress": ("2*lambda*M*(Q:H)-lambda*(H*M+M*H)+Q*H-H*Q; M=Q+I/3"),
                "distortion_stress": ("-L1*(partial_i Q_kl)*(partial_j Q_kl)"),
                "active_stress": "beta*alpha*Q; beta=-1 gives -zeta*Q",
                "molecular_field_in_stress": "raw_H_not_H_over_gamma",
                "molecular_field_linear_space": (execution.molecular_field_linear_space),
                "stress_divergence_sum_space": (execution.stress_divergence_sum_space),
                "pointwise_execution": execution.pointwise_execution,
                "isotropic_stress": "absorbed_into_incompressible_pressure",
                "viscous_stress": "handled_by_eta_laplacian_in_stokes_operator",
            },
            "parameters": {
                "activity_number": preset.activity_number,
                "zeta": zeta,
                "frank_K": frank_k,
                "ldg_A": source.model.ldg_a,
                "ldg_B": source.model.ldg_b,
                "ldg_C": source.model.ldg_c,
                "ldg_L1": ldg_l1,
                "rotational_viscosity_gamma": rotational_viscosity,
                "flow_alignment_lambda": alignment_parameter,
                "alpha": alpha_value,
                "beta": beta,
                "S_initial": initial.initial_s,
                "S_bulk": S_bulk,
                "fric": fric,
                "eta": eta,
            },
        },
        "boundary_conditions": {
            "Q": q_physical_boundary_conditions,
            "Q_assignment": [
                _q_assignment(component, q_physical_boundary_conditions)
                for component in Q_COMPONENTS
            ],
            "Q_evolved_representation": "physical_field",
            "Q_evolved_boundary_conditions": q_evolved_boundary_conditions,
            "velocity_tangential": u_tangential_bc,
            "velocity_normal": u_normal_bc,
            "pressure": pressure_modal_bc,
            "velocity_wall_model_note": (
                "homogeneous kinematic free-slip; not zero total nematic traction"
            ),
        },
        "numerics": {
            "dealiasing": {
                "rule": dealias_rule,
                "fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
                "projected_transform_execution": (projected_transform_execution_metadata),
                "nematic_force_evaluation": (
                    "project_H_then_algebraic_and_distortion_stresses_then_force"
                ),
                "monolithic_quintic_galerkin": False,
                "distortion_stress_z_parity": {
                    "even_components": ["xx", "xy", "yx", "yy", "zz"],
                    "odd_components": ["xz", "yz", "zx", "zy"],
                    "even_basis": q_physical_boundary_conditions,
                    "odd_basis": distortion_odd_z_bc,
                },
            },
            "velocity_zero_mode": zero_mode_policy,
            "zero_mode_force": "total_nematic_tangential_force",
            "pressure_solver": "free_slip_modal_schur_complement",
            "pressure_residual_diagnostics": diagnostics_enabled,
            "molecular_field_linear_space": (execution.molecular_field_linear_space),
            "stress_divergence_sum_space": (execution.stress_divergence_sum_space),
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
                    numerics.hermitian_axis if spectral_storage == "hermitian_half" else None
                ),
                "basis_and_normalization_changed": False,
                "projected_transform_execution": (projected_transform_execution),
            },
            "precision": {
                "real_dtype": real_dtype_name,
                "spectral_dtype": spectral_dtype_name,
                **tf32_metadata,
            },
            "spectral_refresh": {
                "mode": refresh.mode,
                "requested_interval_time": (refresh.requested_interval_time),
                "requested_interval_steps": (refresh.requested_interval_steps),
                "effective_interval_steps": (refresh.effective_interval_steps),
                "effective_interval_time": (refresh.effective_interval_time),
                "phase_origin_step": 0,
            },
        },
        "benchmark_target": "Shendruk et al. PRE 98, 010601(R) (2018), Fig. 4",
        "reproduction_status": "development_reusable_beris_edwards_stokes_stage",
        "scan_variable": "activity_number",
        "parameterization": preset.parameterization,
        "activity_number": preset.activity_number,
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
        "molecular_field_linear_space": (execution.molecular_field_linear_space),
        "stress_divergence_sum_space": (execution.stress_divergence_sum_space),
        "pointwise_execution": execution.pointwise_execution,
        "tf32": construction.tf32,
        "q_boundary_conditions": q_physical_boundary_conditions,
        "q_evolved_boundary_conditions": q_evolved_boundary_conditions,
        "tangential_velocity_boundary_conditions": u_tangential_bc,
        "normal_velocity_boundary_conditions": u_normal_bc,
        "velocity_wall_model": "free-slip",
        "q_wall_model_note": q_wall_model_note,
        "zero_mode_policy": zero_mode_policy,
        "dealias_rule": dealias_rule,
        "dealias_fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
        "projected_transform_execution": projected_transform_execution,
        "ldg_coefficients": {
            "A": source.model.ldg_a,
            "B": source.model.ldg_b,
            "C": source.model.ldg_c,
        },
        "gamma": source.model.gamma,
        "flow_alignment": alignment_parameter,
        "eta": eta,
        "friction": fric,
        "active_stress_beta": beta,
        "S_initial": initial.initial_s,
        "S_bulk": S_bulk,
        "initial_condition": {
            "name": "extruded_analytic_periodic_defect_gas_2d",
            "source": "PSSolver constructed",
            "paper_identical": False,
            "seed": seed,
            "S_initial": initial.initial_s,
            "twist_amplitude": initial.twist_amplitude,
            "twist_modes": initial.twist_modes,
        },
        "initial_defect_gas": {
            "num_pairs": initial.num_defect_pairs,
            "minimum_separation": initial.defect_min_separation,
            "core_radius": initial.defect_core_radius,
            "S_initial": initial.initial_s,
            "background_angle": initial.background_angle,
        },
        "initial_neumann_twist": {
            "rms_amplitude_radians": initial.twist_amplitude,
            "dct_modes": initial.twist_modes,
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
            ("initial condition is PSSolver-constructed and not verified paper-identical"),
            (
                "staged spectral filtering differs from the paper's "
                "finite-difference/LB discretization"
            ),
        ],
    }


def build_run_metadata(compiled, *, normalization=None, tf32_metadata=None):
    """Build the shared pre-run document for the emitted and written metadata."""
    source = compiled.source
    kind = compiled.plan.geometry.kind
    plan_hash = compiled.plan_identity_sha256
    declaration_hash = compiled.declaration_sha256
    requested = compiled._run.construction.device
    normalization = (
        execution_device_normalization(requested) if normalization is None else normalization
    )
    metadata = {
        "schema_version": 3 if kind == "plane_slab" else 2,
        "application": _APPLICATIONS[kind],
        "configuration": source.to_metadata(),
        "declaration_sha256": declaration_hash,
        "plan_identity_sha256": plan_hash,
        "normalization": normalization,
    }
    if kind == "plane_slab":
        if tf32_metadata is None:
            construction = compiled._run.construction
            tf32_metadata = configure_tf32_execution(
                construction.tf32,
                real_dtype=getattr(torch, construction.dtype),
                device=normalization["resolved_execution_device"],
            )
        metadata.update(
            _plane_metadata(
                compiled,
                normalization=normalization,
                tf32_metadata=tf32_metadata,
            )
        )
        initial = metadata["initial_condition"]
        restart = source.output.restart_from
        initial.update(
            artifacts_written=restart is None,
            runtime_construction_only=restart is not None,
            trajectory_state_source=(
                "generated_initial_condition" if restart is None else "authenticated_checkpoint"
            ),
        )
        if restart is not None:
            initial["checkpoint_path"] = str(restart.resolve())
    else:
        metadata.update(
            configuration_sha256=declaration_hash,
            runtime_identity_sha256=plan_hash,
            pressure_gauge="zero_mean",
        )
        if kind == "periodic_box":
            stokes = compiled.plan.constraints[0].arguments
            policy = stokes["zero_mode_policy"]
            metadata.update(
                tangential_zero_mode_policy=policy,
                friction=stokes["friction"],
                uniform_velocity_mode_action=(
                    "remove_all_uniform_velocity_and_force"
                    if policy == "zero_mean"
                    else "retain_all_uniform_velocity_resolved_by_friction"
                ),
            )
        else:
            metadata["velocity_nullspace"] = "absent_due_to_two_bounded_no_slip_axes"
    return metadata


def generate_plane_initial(parameters, geometry, *, dtype, restart):
    """Generate Plane's 2-D precursor and extruded Q in the established order."""
    nx, ny, nz = geometry.shape
    lx, ly, _ = geometry.lengths
    seed = parameters["seed"]
    q_2d = create_initial_condition(
        "analytic_periodic_defect_gas_2d",
        shape=(nx, ny),
        lengths=(lx, ly),
        num_defect_pairs=parameters["num_defect_pairs"],
        min_separation=parameters["defect_min_separation"],
        core_radius=parameters["defect_core_radius"],
        seed=seed,
        S_initial=parameters["initial_s"],
        background_angle=parameters["background_angle"],
        dtype=dtype,
    )
    defects = None
    if not restart:
        defects = sample_periodic_neutral_defects_2d(
            lengths=(lx, ly),
            num_defect_pairs=parameters["num_defect_pairs"],
            min_separation=parameters["defect_min_separation"],
            seed=seed,
        )
    q_initial = create_initial_condition(
        "extruded_2d_twist",
        shape=(nx, ny, nz),
        Q_2d=q_2d,
        boundary_conditions=_Q_BOUNDARY,
        seed=seed,
        twist_amplitude=parameters["twist_amplitude"],
        twist_modes=parameters["twist_modes"],
        dtype=dtype,
    )
    return {name: q_initial[name] for name in Q_COMPONENTS}, q_2d, defects


def write_plane_initial_files(directory, q_2d, defects):
    """Write the old Plane precursor and defect artifact bytes."""
    positions, charges = defects
    directory.mkdir(parents=True, exist_ok=True)
    np.save(
        directory / "Q2D_initial.npy",
        np.stack([q_2d[name].numpy() for name in Q_COMPONENTS], axis=-1),
    )
    np.savetxt(
        directory / "Q2D_defects.csv",
        np.column_stack((positions, charges)),
        delimiter=",",
        header="x,y,charge",
        comments="",
    )


def _require_empty_output(path, kind):
    if not path.exists() or not any(path.iterdir()):
        return
    if kind == "plane_slab":
        raise FileExistsError(f"Refusing to mix pilot outputs in nonempty {path}")
    label = "periodic" if kind == "periodic_box" else "Channel"
    raise FileExistsError(f"refusing nonempty {label} output: {path}")


def _restart_input(compiled, dtype, legacy_preflight):
    path = compiled._run.output.restart_from
    if path is None:
        return 0, None, False
    carrier = compiled._run
    kind = compiled.plan.geometry.kind
    metadata = read_checkpoint_json(
        path / "checkpoint.json", operation=_CHECKPOINT_OPERATIONS[kind]
    )
    if metadata.get("format_kind") != "engine_state":
        if legacy_preflight is None:
            raise ValueError("a v0.4 checkpoint requires a legacy restart preflight")
        start_step, restore = legacy_preflight(dtype)
        return start_step, restore, False
    header = engine_state.read_header(
        path,
        plan_identity_sha256=compiled.plan_identity_sha256,
        dtype=dtype,
        batch=carrier.construction.batch,
    )
    return header["completed_steps"], None, True


def _capture(compiled, engine, kind, checkpoint_diagnostics):
    hydrodynamics = compiled._run.output.save_hydrodynamics
    if kind == "periodic_box":
        return PeriodicCapture(hydrodynamics=hydrodynamics)
    if kind == "rectangular_channel":
        return ChannelCapture(
            engine.projector,
            hydrodynamics=hydrodynamics,
            checkpoint_diagnostics=checkpoint_diagnostics,
        )
    model = compiled.source.model
    if hasattr(model, "eta"):
        viscosity = model.eta
        friction = model.friction_mode_fric if model.zero_mode_policy == "friction" else 0.0
    else:
        viscosity, friction = model.viscosity, model.friction
    return PlaneCapture(
        hydrodynamics=hydrodynamics,
        viscosity=viscosity,
        friction=friction,
        runtime_path="engine",
    )


def tensor_sha256(tensors):
    """Hash an ordered collection of contiguous CPU tensor byte strings."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _record_prepared_metadata(compiled, metadata, engine, state, values, start_step, is_engine):
    kind = compiled.plan.geometry.kind
    device = metadata["normalization"]["resolved_execution_device"]
    metadata["runtime_construction"] = {
        "device": device,
        "plan_identity_sha256": compiled.plan_identity_sha256,
    }
    if kind == "plane_slab":
        fields = engine.field_view(state)
        initial = metadata["initial_condition"]
        initial["raw_q_sha256"] = tensor_sha256(values[name] for name in Q_COMPONENTS)
        initial["projected_q_sha256"] = tensor_sha256(fields[name] for name in Q_COMPONENTS)
        metadata["retained_q_modes"] = engine.projector.retained_axis_counts(fields["Qxx.bc"])
        metadata["retained_normal_velocity_modes"] = engine.projector.retained_axis_counts(
            fields["uz.bc"],
        )
        source = compiled._run.output.restart_from
        metadata["restart"] = {"kind": "fresh"}
        if source is not None:
            metadata["restart"] = {
                "kind": "engine_state" if is_engine else "same_backend_plane_workflow_checkpoint",
                "source": str(source.resolve()),
                "source_completed_steps": start_step,
            }
            if is_engine:
                metadata["restart"]["source_runtime_identity_sha256"] = (
                    compiled.plan_identity_sha256
                )
            initial["used_for_runtime_construction_only"] = True
        metadata["workflow"] = {
            "schema_version": 1,
            "checkpoint_format_version": "engine_state.v1",
            "cross_backend_restart_supported": True,
            "start_step": start_step,
            "requested_additional_steps": compiled._run.output.steps,
            "completion_marker_order": "metadata_then_COMPLETE",
        }
    elif kind == "periodic_box":
        metadata["runtime"] = {
            key: metadata[key]
            for key in (
                "pressure_gauge",
                "tangential_zero_mode_policy",
                "friction",
                "uniform_velocity_mode_action",
            )
        }
        metadata["runtime"]["hermitian_state_projection"] = engine.projector.execution_metadata()[
            "hermitian_state_projection"
        ]
    else:
        metadata["runtime"] = {
            "pressure_gauge": metadata["pressure_gauge"],
            "velocity_nullspace": metadata["velocity_nullspace"],
            "pressure_warm_start": True,
            "pressure_diagnostics_schema_version": 2,
            "pressure_solver_policy": engine.constraint.term.pressure_solver.pressure_convergence_policy(),
        }


def prepare_run(compiled, *, metadata, device, dtype, legacy_preflight=None, initial_values=None):
    """Prepare a run; only qualification supplies the explicit initial values."""
    carrier = compiled._run
    plan = carrier.assembled.plan
    kind = plan.geometry.kind
    output = carrier.output
    path = output.directory.expanduser().resolve()
    start_step, restore, is_engine_checkpoint = _restart_input(compiled, dtype, legacy_preflight)
    if output.save_start_step > start_step + output.steps:
        raise ValueError("save_start_step must not exceed the absolute final step")

    plane_files = None
    if initial_values is not None:
        _require_empty_output(path, kind)
        values = initial_values
    elif kind == "plane_slab":
        _require_empty_output(path, kind)
        values, q_2d, defects = generate_plane_initial(
            carrier.initial_condition.parameters,
            plan.geometry,
            dtype=dtype,
            restart=output.restart_from is not None,
        )
        if defects is not None:
            plane_files = q_2d, defects
    else:
        label = "periodic" if kind == "periodic_box" else "Channel"
        values, snapshot, digest = load_q_snapshot(
            carrier.initial_condition.directory,
            carrier.initial_condition.step,
            shape=plan.geometry.shape,
            dtype=carrier.construction.dtype,
            components=Q_COMPONENTS,
            label=label,
            application_family=_APPLICATIONS[kind],
        )
        metadata["initial_condition"] = {
            "source": str(snapshot),
            "sha256": digest,
            "logical_mode": carrier.initial_condition.mode,
        }
        _require_empty_output(path, kind)
    engine = carrier.assembled.engine(device=device, dtype=dtype, batch=carrier.construction.batch)
    initial = {name: value.to(device=device) for name, value in values.items()}
    construction_diagnostics = {}
    state = engine.initial_state(initial, diagnostics=construction_diagnostics)
    diagnostics = dict(construction_diagnostics) if kind == "rectangular_channel" else {}
    # The Plane initializer's hashes describe construction, not the restored state.
    _record_prepared_metadata(
        compiled, metadata, engine, state, values, start_step, is_engine_checkpoint
    )
    if plane_files is not None:
        write_plane_initial_files(path, *plane_files)
    if is_engine_checkpoint:
        state = engine_state.read(
            output.restart_from,
            plan_identity_sha256=plan.identity_sha256,
            dtype=dtype,
            batch=carrier.construction.batch,
            device=device,
        )
        if kind == "rectangular_channel":
            engine._restore_channel_state_layout(state)
        if kind == "plane_slab":
            state.constrained_current = False
            engine.prepare_constraints_(state, diagnostics=diagnostics)
    elif restore is not None:
        start_step, legacy_identity = restore(engine, state, diagnostics)
        if kind == "plane_slab":
            metadata["restart"]["source_runtime_identity_sha256"] = legacy_identity

    def checkpoint_diagnostics():
        return dict(runner.constraint_diagnostics)

    runner = Runner(
        engine,
        schedule=RunSchedule(
            output.steps,
            output.save_interval,
            output.save_start_step,
            output.diagnostics,
            output.diagnostic_interval,
            output.checkpoint_interval,
        ),
        output_directory=path,
        capture=_capture(compiled, engine, kind, checkpoint_diagnostics),
        metadata=metadata,
    )
    if kind == "rectangular_channel" or (
        kind == "plane_slab" and (is_engine_checkpoint or restore is not None)
    ):
        runner.constraint_diagnostics.update(diagnostics)
    return _PreparedRun(engine, state, runner, start_step, construction_diagnostics)


def run_assembled_simulation(
    compiled,
    *,
    progress=None,
    emit_metadata=False,
    legacy_preflight=None,
):
    """Run the already compiled Plan without compiling a second time."""
    construction = compiled._run.construction
    if compiled._run.output.dry_run:
        path = compiled._run.output.restart_from
        if path is not None:
            try:
                header = read_checkpoint_json(
                    path / "checkpoint.json",
                    operation="engine_state_import",
                )
            except CheckpointError:
                header = {}
            engine_state._reject_test_integrator(header, validate_seal=True)
        metadata = build_run_metadata(compiled)
        if emit_metadata:
            print(json.dumps(metadata, indent=2, sort_keys=True))
        return None
    normalization = execution_device_normalization(construction.device)
    device = resolve_execution_device(normalization["resolved_execution_device"])
    dtype = getattr(torch, construction.dtype)
    with scoped_tf32_execution(construction.tf32, real_dtype=dtype, device=device) as precision:
        metadata = build_run_metadata(
            compiled,
            normalization=normalization,
            tf32_metadata=precision,
        )
        if emit_metadata:
            print(json.dumps(metadata, indent=2, sort_keys=True))
        if compiled.plan.geometry.kind != "plane_slab":
            metadata["precision"] = {
                "real_dtype": construction.dtype,
                "spectral_dtype": "complex64" if dtype == torch.float32 else "complex128",
                **precision,
            }
        prepared = prepare_run(
            compiled,
            metadata=metadata,
            device=device,
            dtype=dtype,
            legacy_preflight=legacy_preflight,
        )
        return prepared.runner.run(
            prepared.state, start_step=prepared.start_step, progress=progress
        )
