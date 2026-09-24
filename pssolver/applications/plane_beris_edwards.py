"""Supported Plane Beris--Edwards--Stokes application.

This module owns the high-level application assembly that used to live in the
repository-root compatibility script.  Importing it never parses command-line
arguments or starts a simulation.

The modeled equations and limitations remain:

This application descends from ``Plane_shendruk_stokes.py``. It retains the
Shendruk benchmark parameterization and mixed-basis geometry while using the
reusable PSSolver Beris--Edwards Q model for the equation

    (partial_t + u dot grad) Q - S(E, Omega, Q) = H / gamma

with the full co-rotational flow-alignment term and the one-constant
Landau--de Gennes molecular field. The quasistatic momentum equation is

    0 = -grad(p) + eta lap(u) - fric u + div(Pi_nem),

where, up to an isotropic term absorbed into pressure,

    Pi_nem = 2 lambda M (Q:H) - lambda (H M + M H)
             + Q H - H Q - L1 partial_i Q:partial_j Q
             + beta alpha Q,                 M = Q + I/3.

Thus beta=-1 and alpha=zeta give the paper's active stress -zeta Q while the
reactive and one-constant distortion stresses are retained as well.

The independent scan variable is the paper's dimensionless activity number

    A = H * sqrt(zeta / K),

where H is the channel height and K is the one-constant Frank elastic
constant. This file runs exactly one value of A using the mixed FFT/DCT/DST
free-slip architecture from Plane_free_slip_dealiased.py:

* Q and tangential velocity use Neumann/DCT modes at the z walls;
* normal velocity uses Dirichlet/DST modes at the z walls;
* pressure is a Neumann/DCT modal incompressibility multiplier;
* the default plug-flow convention is the zero-mean pseudoinverse;
* the default no-padding dealiasing rule is cubic_half.

This is not yet a strict Fig. 4 reproduction: zero-Reynolds-number Stokes
replaces the paper's inertial momentum dynamics, and the initialization and
spectral discretization are not paper-identical.
"""

from collections.abc import Iterable, Sequence
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
    PlaneRuntimePath,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.workflows import (
    PlaneBerisEdwardsWorkflow,
    PlaneWorkflowResult,
    read_plane_checkpoint_header,
)
from pssolver.models.active_nematics import (
    BerisEdwardsPointwiseKernels,
    Q_convention_metadata,
    create_initial_condition,
    sample_periodic_neutral_defects_2d,
)
from pssolver.operators.projection import (
    DEALIAS_RULE_FRACTIONS,
)
from tqdm import trange


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


def parse_args(argv: Sequence[str] | None = None) -> PlaneBerisEdwardsRunSpec:
    """Delegate the production CLI to its immutable run specification."""

    return parse_plane_beris_edwards_run_spec(argv)


def tensor_sha256(tensors):
    """Hash an ordered collection of contiguous CPU tensor byte strings."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_plane_beris_edwards(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
) -> PlaneWorkflowResult | None:
    """Execute one validated Plane run through the supported application API.

    A dry-run returns None after resolving and optionally printing metadata.
    A simulation returns its PlaneWorkflowResult. The caller owns progress
    presentation; omitting progress selects the quiet consecutive range.
    """

    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PlaneBerisEdwardsRunSpec")
    components = decompose_plane_beris_edwards_run_spec(run_spec)
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

    seed = initial_condition.seed
    dt = time_stepping.dt
    steps = workflow_spec.steps
    requested_device = execution.device
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    real_dtype_name = numerics.precision.value
    real_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[real_dtype_name]
    spectral_dtype_name = "complex64" if real_dtype == torch.float32 else "complex128"
    tf32_requested = execution.tf32 == "on"
    tf32_effective = (
        tf32_requested
        and real_dtype == torch.float32
        and torch.device(device).type == "cuda"
    )
    torch.set_float32_matmul_precision("high" if tf32_effective else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32_effective
    torch.backends.cudnn.allow_tf32 = tf32_effective

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
        "schema_version": 1,
        "script": "Plane_beris_edwards_stokes.py",
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
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
            "Q": Q_BC,
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
                "tf32_requested": execution.tf32,
                "tf32_effective": tf32_effective,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
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
        "q_boundary_conditions": Q_BC,
        "tangential_velocity_boundary_conditions": U_TANGENTIAL_BC,
        "normal_velocity_boundary_conditions": U_NORMAL_BC,
        "velocity_wall_model": "free-slip",
        "q_wall_model_note": "free/Neumann; matches the Fig. 4 free-anchoring branch",
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
    if emit_metadata:
        print(json.dumps(metadata, indent=2))
    if invocation.dry_run:
        return None

    # Reject unsupported cross-runtime or numerically incompatible restart before
    # creating an output directory, generating Q, or constructing either solver.
    if workflow_spec.restart_from is not None:
        restart_header = read_plane_checkpoint_header(workflow_spec.restart_from)
        if restart_header.runtime_path is not execution.runtime_path:
            raise ValueError(
                "cross-runtime Plane checkpoint restart is unsupported"
            )
        if (
            restart_header.runtime_identity_sha256
            != run_spec.runtime_identity_sha256()
        ):
            raise ValueError("checkpoint runtime identity does not match target")

    output_dir = workflow_spec.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix pilot outputs in nonempty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    defect_positions, defect_charges = sample_periodic_neutral_defects_2d(
        lengths=(Lx, Ly),
        num_defect_pairs=initial_condition.num_defect_pairs,
        min_separation=initial_condition.defect_min_separation,
        seed=seed,
    )
    np.save(
        output_dir / "Q2D_initial.npy",
        np.stack(
            [q_2d[name].numpy() for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")],
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
    runtime_request = PlaneRuntimeBuildRequest(
        run_spec=run_spec,
        production_metadata=metadata,
        initial_values=initial_values,
        device=device,
    )
    if execution.runtime_path is PlaneRuntimePath.SEPARATED_CANARY:
        metadata["runtime_construction"] = {
            "schema_version": 1,
            "owner": "pssolver.runtime.plane_beris_edwards",
            "compatibility_exception": "separated_canary",
            "package_plan": None,
            "fallback_allowed": False,
        }
        runtime_adapter = build_plane_beris_edwards_runtime(runtime_request)
    else:
        construction = PackageRuntimeConstructionInput(
            plan=plan_package_runtime_construction(
                compose_plane_beris_edwards_simulation(components)
            ),
            request=runtime_request,
        )
        metadata["runtime_construction"] = {
            "plan": construction.plan.to_metadata(),
            "input": construction.to_metadata(),
        }
        runtime_adapter = build_package_simulation_runtime(construction)
    spectral_projector = runtime_adapter.projector
    metadata["runtime_selection"] = {
        **run_spec.runtime_selection_metadata(),
        **runtime_adapter.to_metadata(),
    }
    metadata["initial_condition"]["projected_q_sha256"] = tensor_sha256(
        tuple(
            runtime_adapter.fields[name]
            for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
        )
    )
    metadata["retained_q_modes"] = spectral_projector.retained_axis_counts(Q_BC)
    metadata["retained_normal_velocity_modes"] = (
        spectral_projector.retained_axis_counts(U_NORMAL_BC)
    )
    workflow = PlaneBerisEdwardsWorkflow(
        runtime_adapter,
        run_spec,
        output_dir,
        metadata,
    )
    workflow_result = workflow.run(progress=progress)
    return workflow_result


def _print_workflow_summary(workflow_result: PlaneWorkflowResult) -> None:
    print(f"Elapsed time: {workflow_result.elapsed_seconds:.6f} seconds")
    if workflow_result.diagnostics:
        final_diagnostic = workflow_result.diagnostics[-1]
        print(
            "Final div(u) diagnostic: "
            f"max={final_diagnostic.div_max:.6e}, "
            f"rms={final_diagnostic.div_rms:.6e}, "
            f"relative={final_diagnostic.div_rel:.6e}"
        )
        print(
            "Final Schur saddle solve diagnostic: "
            f"iterations={int(final_diagnostic.schur_iterations)}, "
            f"abs_residual={final_diagnostic.schur_abs_residual:.6e}, "
            f"rel_residual={final_diagnostic.schur_rel_residual:.6e}"
        )
        print(
            "Final wall normal momentum diagnostic: "
            f"max={final_diagnostic.wall_normal_momentum_max:.6e}, "
            f"rms={final_diagnostic.wall_normal_momentum_rms:.6e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical production CLI without import-time side effects."""

    run_spec = parse_args(argv)
    components = decompose_plane_beris_edwards_run_spec(run_spec)
    progress = (
        None
        if components.invocation.dry_run
        else trange(components.workflow.steps)
    )
    result = run_plane_beris_edwards(
        run_spec,
        progress=progress,
        emit_metadata=True,
    )
    if result is not None:
        _print_workflow_summary(result)
    return 0


__all__ = [
    "main",
    "parse_args",
    "run_plane_beris_edwards",
]
