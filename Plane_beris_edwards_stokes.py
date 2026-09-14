"""Reusable Beris--Edwards model with complete-stress Stokes flow.

This independent script is copied from Plane_shendruk_stokes.py. It retains
the Shendruk benchmark parameterization and mixed-basis geometry while using
the reusable PSSolver Beris--Edwards Q model for the equation

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

import hashlib
import json
import platform
from pathlib import Path
import time
import torch
from pssolver import SpectralSolver, write_run_metadata
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PLANE_FREE_SLIP_BOUNDARIES,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    BerisEdwardsPointwiseKernels,
    Q_convention_metadata,
    beris_edwards_linear_operator,
    create_initial_condition,
    sample_periodic_neutral_defects_2d,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.plane import PLANE_HERMITIAN_AXIS
from pssolver.transforms import (
    BasisAwareSpectralProjector,
    DEALIAS_RULE_FRACTIONS,
)
from tqdm import trange
import numpy as np


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
ENABLE_DIAGNOSTICS = False
DIAGNOSTIC_INTERVAL = 100
SAVE_INTERVAL = 500
SAVE_HYDRODYNAMICS = False
ALIGNMENT_PARAMETER = 0.3

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


def parse_args():
    """Delegate the production CLI to its immutable run specification."""

    return parse_plane_beris_edwards_run_spec()


def tensor_sha256(tensors):
    """Hash an ordered collection of contiguous CPU tensor byte strings."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class DealiasedSemiImplicitEulerIntegrator(SemiImplicitEulerIntegrator):
    """Semi-implicit Euler with projection before every inverse transform.

    This intentionally mirrors the core integrator so the nonlinear spectrum
    is truncated after the IMEX update and before Q returns to real space. The
    periodic real-to-spectral refresh is projected and synchronized as well.
    """

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector

    def _refresh_dynamic_spectra(self):
        self.spectral_projector.refresh_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

    def step(self, pre_update_callback=None):
        if self._static_fields_are_current:
            self._static_fields_are_current = False
        else:
            self.model.update_static_fields()

        if pre_update_callback is not None:
            pre_update_callback()

        nonlinear_hats = self.model.compute_nonlinear()
        dynamic_fields = self.model.fields.spectral[:self.dyn_count]
        dynamic_fields.add_(self.dt * nonlinear_hats)
        dynamic_fields.div_(self.denom)
        self.spectral_projector.project_dynamic_fields(
            self.model.fields,
            sync_spatial=False,
        )

        for group in self.dynamic_transform_groups:
            boundary_conditions = self.model.fields.get_boundary_conditions(
                group[0]
            )
            self.model.fields.spatial[group] = (
                self.spectral_projector.inverse_transform(
                    self.model.fields.spectral[group],
                    boundary_conditions,
                )
            )

        self._advance_spectral_refresh_clock()


def divergence_stats(fields):
    div_u = (
        fields.gradient('ux', axis=0)
        + fields.gradient('uy', axis=1)
        + fields.gradient('uz', axis=2)
    )
    div_abs = div_u.abs()
    div_max = div_abs.max().item()
    div_rms = torch.sqrt(torch.mean(div_abs.square())).item()

    gxux = fields.gradient('ux', axis=0)
    gyux = fields.gradient('ux', axis=1)
    gzux = fields.gradient('ux', axis=2)
    gxuy = fields.gradient('uy', axis=0)
    gyuy = fields.gradient('uy', axis=1)
    gzuy = fields.gradient('uy', axis=2)
    gxuz = fields.gradient('uz', axis=0)
    gyuz = fields.gradient('uz', axis=1)
    gzuz = fields.gradient('uz', axis=2)
    grad_u_sq = (
        gxux.abs().square() + gyux.abs().square() + gzux.abs().square()
        + gxuy.abs().square() + gyuy.abs().square() + gzuy.abs().square()
        + gxuz.abs().square() + gyuz.abs().square() + gzuz.abs().square()
    )
    grad_u_rms = torch.sqrt(torch.mean(grad_u_sq)).item()
    div_rel = div_rms / max(grad_u_rms, 1e-30)
    return div_max, div_rms, div_rel


def wall_normal_momentum_stats(fields, static_model):
    """Check normal momentum on the first and last cell-center planes."""
    fz = static_model.last_projected_normal_force
    if fz is None:
        raise RuntimeError(
            "Normal-force diagnostics were not enabled in the static model."
        )

    lap_uz = fields.laplacian('uz')
    dzp = fields.gradient('p', axis=2)
    residual = dzp - (
        fz
        + static_model.viscosity * lap_uz
        - static_model.friction * fields['uz']
    )

    wall_residual = torch.stack([residual[..., 0], residual[..., -1]], dim=-1)
    wall_abs = wall_residual.abs()
    return wall_abs.max().item(), torch.sqrt(torch.mean(wall_abs.square())).item()

args = parse_args()

seed = args.seed
dt = args.dt
steps = args.steps
device = (
    "cuda" if args.device == "auto" and torch.cuda.is_available()
    else "cpu" if args.device == "auto"
    else args.device
)
real_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
}[args.dtype]
spectral_dtype_name = "complex64" if real_dtype == torch.float32 else "complex128"
tf32_requested = args.tf32 == "on"
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
pointwise_kernels = BerisEdwardsPointwiseKernels(args.pointwise_execution)
pointwise_execution_metadata = pointwise_kernels.metadata()
batchsize = 1

Nx, Ny, Nz = args.nx, args.ny, args.nz
Lx, Ly, Lz = args.lx, args.ly, args.height
SAVE_INTERVAL = args.save_interval
SAVE_START_STEP = args.save_start_step
DIAGNOSTIC_INTERVAL = args.diagnostic_interval
ENABLE_DIAGNOSTICS = args.diagnostics
SAVE_HYDRODYNAMICS = args.save_hydrodynamics
ALIGNMENT_PARAMETER = args.flow_alignment
zero_mode_policy = args.zero_mode_policy
dealias_rule = args.dealias_rule
projected_transform_execution_metadata = {
    "requested": args.projected_transform_execution,
    "effective": args.projected_transform_execution,
    "fallback_allowed": False,
    "fallback_reason": None,
    "truncated_real_basis_axes": (
        args.projected_transform_execution == "truncated"
    ),
    "spectral_storage": args.spectral_storage,
    "backend_storage_shape_preserved": True,
    "full_spectral_storage_preserved": (
        args.spectral_storage == "full_complex"
    ),
}
spectral_shape = [
    Nx,
    Ny // 2 + 1 if args.spectral_storage == "hermitian_half" else Ny,
    Nz,
]

# Write Q = q (nn-I/3), where q = 3 S/2 in the declared convention.
# Equating (L1/2) |grad Q|^2 with (K/2) |grad n|^2 gives
# K = 2 L1 q_eq^2 and therefore L1 = K/(2 q_eq^2).  At the Shendruk
# bulk equilibrium S_eq=1/3, q_eq=1/2 and L1=2K.  Using q_eq explicitly
# avoids the scalar-amplitude ambiguity in the paper's printed mapping.
resolved_preset = args.shendruk_preset
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
beta = args.beta
fric = (
    0.0
    if zero_mode_policy == "zero_mean"
    else float(args.friction_mode_fric)
)
eta = args.eta

implementation_provenance = {
    "schema_version": 1,
    "files": {
        relative_path: file_sha256(
            Path(__file__).resolve().parent / relative_path
        )
        for relative_path in IMPLEMENTATION_SOURCE_FILES
    },
}

metadata = {
    "schema_version": 1,
    "script": "Plane_beris_edwards_stokes.py",
    "configuration": args.identity_metadata(),
    "validation_config_sha256": args.validation_config_sha256,
    "implementation_provenance": implementation_provenance,
    "runtime_environment": runtime_environment,
    "solver": {
        "shape": [Nx, Ny, Nz],
        "spectral_shape": spectral_shape,
        "lengths": [Lx, Ly, Lz],
        "dt": dt,
        "steps": steps,
        "save_interval": SAVE_INTERVAL,
        "real_dtype": args.dtype,
        "spectral_dtype": spectral_dtype_name,
        "transform_execution_order": args.transform_execution_order,
        "spectral_storage": args.spectral_storage,
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
            "pointwise_execution": args.pointwise_execution,
            "raw_coefficients": {
                "A": args.ldg_a,
                "B": args.ldg_b,
                "C": args.ldg_c,
                "L1": ldg_l1,
                "rotational_viscosity_gamma": rotational_viscosity,
                "flow_alignment_lambda": ALIGNMENT_PARAMETER,
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
                args.molecular_field_linear_space
            ),
            "stress_divergence_sum_space": (
                args.stress_divergence_sum_space
            ),
            "pointwise_execution": args.pointwise_execution,
            "isotropic_stress": "absorbed_into_incompressible_pressure",
            "viscous_stress": "handled_by_eta_laplacian_in_stokes_operator",
        },
        "parameters": {
            "activity_number": args.activity_number,
            "zeta": zeta,
            "frank_K": frank_k,
            "ldg_A": args.ldg_a,
            "ldg_B": args.ldg_b,
            "ldg_C": args.ldg_c,
            "ldg_L1": ldg_l1,
            "rotational_viscosity_gamma": rotational_viscosity,
            "flow_alignment_lambda": ALIGNMENT_PARAMETER,
            "alpha": alpha_value,
            "beta": beta,
            "S_initial": args.S_initial,
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
        "pressure_residual_diagnostics": ENABLE_DIAGNOSTICS,
        "molecular_field_linear_space": args.molecular_field_linear_space,
        "stress_divergence_sum_space": args.stress_divergence_sum_space,
        "pointwise_kernels": pointwise_execution_metadata,
        "q_gradient_reuse": {
            "enabled": not args.disable_q_gradient_reuse,
            "scope": "single_static_to_nonlinear_evaluation",
            "mutation_guard": "spatial_and_spectral_tensor_versions",
        },
        "transforms": {
            "execution_order": args.transform_execution_order,
            "spectral_storage": args.spectral_storage,
            "physical_shape": [Nx, Ny, Nz],
            "spectral_shape": spectral_shape,
            "hermitian_axis": (
                1 if args.spectral_storage == "hermitian_half" else None
            ),
            "basis_and_normalization_changed": False,
            "projected_transform_execution": (
                args.projected_transform_execution
            ),
        },
        "precision": {
            "real_dtype": args.dtype,
            "spectral_dtype": spectral_dtype_name,
            "tf32_requested": args.tf32,
            "tf32_effective": tf32_effective,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "spectral_refresh": {
            "mode": args.spectral_refresh_mode,
            "requested_interval_time": args.spectral_refresh_requested_time,
            "requested_interval_steps": args.spectral_refresh_requested_steps,
            "effective_interval_steps": args.spectral_refresh_interval_steps,
            "effective_interval_time": args.spectral_refresh_effective_time,
            "phase_origin_step": 0,
        },
    },
    "benchmark_target": "Shendruk et al. PRE 98, 010601(R) (2018), Fig. 4",
    "reproduction_status": "development_reusable_beris_edwards_stokes_stage",
    "scan_variable": "activity_number",
    "parameterization": args.parameterization,
    "activity_number": args.activity_number,
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
    "save_start_step": SAVE_START_STEP,
    "save_interval": SAVE_INTERVAL,
    "diagnostic_interval": DIAGNOSTIC_INTERVAL,
    "seed": seed,
    "device": device,
    "dtype": args.dtype,
    "transform_execution_order": args.transform_execution_order,
    "spectral_storage": args.spectral_storage,
    "molecular_field_linear_space": args.molecular_field_linear_space,
    "stress_divergence_sum_space": args.stress_divergence_sum_space,
    "pointwise_execution": args.pointwise_execution,
    "tf32": args.tf32,
    "q_boundary_conditions": Q_BC,
    "tangential_velocity_boundary_conditions": U_TANGENTIAL_BC,
    "normal_velocity_boundary_conditions": U_NORMAL_BC,
    "velocity_wall_model": "free-slip",
    "q_wall_model_note": "free/Neumann; matches the Fig. 4 free-anchoring branch",
    "zero_mode_policy": zero_mode_policy,
    "dealias_rule": dealias_rule,
    "dealias_fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
    "projected_transform_execution": args.projected_transform_execution,
    "ldg_coefficients": {"A": args.ldg_a, "B": args.ldg_b, "C": args.ldg_c},
    "gamma": args.gamma,
    "flow_alignment": ALIGNMENT_PARAMETER,
    "eta": eta,
    "friction": fric,
    "active_stress_beta": beta,
    "S_initial": args.S_initial,
    "S_bulk": S_bulk,
    "initial_condition": {
        "name": "extruded_analytic_periodic_defect_gas_2d",
        "source": "PSSolver constructed",
        "paper_identical": False,
        "seed": seed,
        "S_initial": args.S_initial,
        "twist_amplitude": args.twist_amplitude,
        "twist_modes": args.twist_modes,
    },
    "initial_defect_gas": {
        "num_pairs": args.num_defect_pairs,
        "minimum_separation": args.defect_min_separation,
        "core_radius": args.defect_core_radius,
        "S_initial": args.S_initial,
        "background_angle": args.background_angle,
    },
    "initial_neumann_twist": {
        "rms_amplitude_radians": args.twist_amplitude,
        "dct_modes": args.twist_modes,
    },
    "save_hydrodynamics": SAVE_HYDRODYNAMICS,
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
        "initial condition is PSSolver-constructed and not verified paper-identical",
        (
            "staged spectral filtering differs from the paper's "
            "finite-difference/LB discretization"
        ),
    ],
}
print(json.dumps(metadata, indent=2))
if args.dry_run:
    raise SystemExit(0)

output_dir = args.output_dir.resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(f"Refusing to mix pilot outputs in nonempty {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

q_2d = create_initial_condition(
    "analytic_periodic_defect_gas_2d",
    shape=(Nx, Ny),
    lengths=(Lx, Ly),
    num_defect_pairs=args.num_defect_pairs,
    min_separation=args.defect_min_separation,
    core_radius=args.defect_core_radius,
    seed=seed,
    S_initial=args.S_initial,
    background_angle=args.background_angle,
    dtype=real_dtype,
)
defect_positions, defect_charges = sample_periodic_neutral_defects_2d(
    lengths=(Lx, Ly),
    num_defect_pairs=args.num_defect_pairs,
    min_separation=args.defect_min_separation,
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

solver = SpectralSolver(
    shape=(Nx, Ny, Nz),
    L=(Lx, Ly, Lz),
    dt=dt,
    device=device,
    batchsize=batchsize,
    dtype=real_dtype,
    transform_execution_order=args.transform_execution_order,
    spectral_storage=args.spectral_storage,
    hermitian_axis=PLANE_HERMITIAN_AXIS,
)
spectral_projector = BasisAwareSpectralProjector(
    solver,
    rule=dealias_rule,
    transform_execution=args.projected_transform_execution,
)
solver.model.spectral_projector = spectral_projector
solver.model.set_static_inverse_transform(
    spectral_projector.inverse_transform
)
solver.integrator_cl = DealiasedSemiImplicitEulerIntegrator
q_initial_condition = create_initial_condition(
    "extruded_2d_twist",
    shape=(Nx, Ny, Nz),
    Q_2d=q_2d,
    boundary_conditions=Q_BC,
    seed=seed,
    twist_amplitude=args.twist_amplitude,
    twist_modes=tuple(args.twist_modes),
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

q2_Q = solver.get_q2(Q_BC)
q_linear_operator = beris_edwards_linear_operator(
    q2_Q,
    ldg_a=args.ldg_a,
    ldg_l1=ldg_l1,
    rotational_viscosity=rotational_viscosity,
)

# --- Add active fields ---
solver.model.add_dynamic_field(
    "Qxx",
    init = Qxx_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qxy",
    init =  Qxy_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qxz",
    init = Qxz_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qyy",
    init =  Qyy_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qyz",
    init =  Qyz_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)

# --- Add static fields ---
# Free-slip mixed modal spaces: tangential velocity uses DCT in z, normal
# velocity uses DST in z, and pressure is the DCT incompressibility multiplier.
solver.model.add_static_field("ux", boundary_conditions=U_TANGENTIAL_BC)
solver.model.add_static_field("uy", boundary_conditions=U_TANGENTIAL_BC)
solver.model.add_static_field("uz", boundary_conditions=U_NORMAL_BC)
solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)

q_gradient_cache = (
    None
    if args.disable_q_gradient_reuse
    else BerisEdwardsQGradientCache()
)
solver.model.set_nonlinear_model(
    BerisEdwardsQNonlinearModel(
        spectral_projector,
        Q_BC,
        ldg_b=args.ldg_b,
        ldg_c=args.ldg_c,
        rotational_viscosity=rotational_viscosity,
        flow_alignment=ALIGNMENT_PARAMETER,
        q_gradient_cache=q_gradient_cache,
        pointwise_kernels=pointwise_kernels,
    )
)
solver.model.set_static_compute_model(
    BerisEdwardsFreeSlipStokes(
        solver,
        spectral_projector=spectral_projector,
        beta_value=beta,
        friction=fric,
        viscosity=eta,
        ldg_a=args.ldg_a,
        ldg_b=args.ldg_b,
        ldg_c=args.ldg_c,
        ldg_l1=ldg_l1,
        flow_alignment=ALIGNMENT_PARAMETER,
        molecular_field_linear_space=args.molecular_field_linear_space,
        stress_divergence_sum_space=args.stress_divergence_sum_space,
        cache_force_diagnostics=ENABLE_DIAGNOSTICS,
        cache_pressure_diagnostics=ENABLE_DIAGNOSTICS,
        q_gradient_cache=q_gradient_cache,
        pointwise_kernels=pointwise_kernels,
        zero_mode_policy=zero_mode_policy,
    )
)

alpha = torch.tensor(alpha_value, device=device, dtype=real_dtype)
solver.model.parameters.new_param('alpha', alpha)

solver.build()
solver.integrator.set_spectral_refresh_interval(
    args.spectral_refresh_interval_steps
)

# Band-limit the generated Q field before it participates in pointwise products.
spectral_projector.project_dynamic_fields(
    solver.model.fields,
    sync_spatial=True,
)
if spectral_projector.enabled:
    solver.integrator._static_fields_are_current = False
metadata["initial_condition"]["projected_q_sha256"] = tensor_sha256(
    tuple(
        solver.model.fields[name]
        for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    )
)
metadata["retained_q_modes"] = spectral_projector.retained_axis_counts(Q_BC)
metadata["retained_normal_velocity_modes"] = spectral_projector.retained_axis_counts(
    U_NORMAL_BC
)
write_run_metadata(output_dir, metadata, status="running")

start_step = 0

diagnostic_history = []
start = time.time()
pbar = trange(steps)


def save_snapshot(i):
    q_snapshot = torch.stack([
        solver.model.fields[name].detach().cpu()
        for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
    ])  # shape -> (5, batch, Nx, Ny, Nz)
    q_snapshot = q_snapshot.permute(1, 2, 3, 4, 0)
    np.save(output_dir / f"Q_{i}.npy", q_snapshot[0].numpy())

    if SAVE_HYDRODYNAMICS:
        u_snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["ux", "uy", "uz"]
        ])  # shape -> (3, batch, Nx, Ny, Nz)
        u_snapshot = u_snapshot.permute(1, 2, 3, 4, 0)
        np.save(output_dir / f"u_{i}.npy", u_snapshot[0].numpy())

        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(output_dir / f"p_{i}.npy", p_snapshot[0].numpy())


def record_step_state(i):
    if ENABLE_DIAGNOSTICS and i % DIAGNOSTIC_INTERVAL == 0:
        div_max, div_rms, div_rel = divergence_stats(solver.model.fields)
        static_model = solver.model.static_model
        wall_mom_max, wall_mom_rms = wall_normal_momentum_stats(
            solver.model.fields,
            static_model,
        )
        diagnostic_history.append((
            i,
            div_max,
            div_rms,
            div_rel,
            static_model.last_pressure_iterations,
            static_model.last_pressure_residual,
            static_model.last_pressure_relative_residual,
            wall_mom_max,
            wall_mom_rms,
        ))
        pbar.set_postfix(
            div_max=f"{div_max:.2e}",
            div_rms=f"{div_rms:.2e}",
            div_rel=f"{div_rel:.2e}",
            schur_it=static_model.last_pressure_iterations,
            schur_rel=f"{static_model.last_pressure_relative_residual:.2e}",
            wall_n_rms=f"{wall_mom_rms:.2e}",
        )
    if i >= SAVE_START_STEP and i % SAVE_INTERVAL == 0:
        save_snapshot(i)


for local_step in pbar:
    i = start_step + local_step
    solver.run(1, pre_update_callback=lambda _solver, _step, i=i: record_step_state(i))

final_step = start_step + steps
solver.refresh_static_fields()
save_snapshot(final_step)

end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
if ENABLE_DIAGNOSTICS:
    final_div_max, final_div_rms, final_div_rel = divergence_stats(solver.model.fields)
    static_model = solver.model.static_model
    final_wall_mom_max, final_wall_mom_rms = wall_normal_momentum_stats(
        solver.model.fields,
        static_model,
    )
    diagnostic_history.append((
        start_step + steps,
        final_div_max,
        final_div_rms,
        final_div_rel,
        static_model.last_pressure_iterations,
        static_model.last_pressure_residual,
        static_model.last_pressure_relative_residual,
        final_wall_mom_max,
        final_wall_mom_rms,
    ))
    diagnostic_array = np.array(
        diagnostic_history,
        dtype=[
            ("step", np.int64),
            ("div_max", np.float64),
            ("div_rms", np.float64),
            ("div_rel", np.float64),
            ("schur_iterations", np.float64),
            ("schur_abs_residual", np.float64),
            ("schur_rel_residual", np.float64),
            ("wall_normal_momentum_max", np.float64),
            ("wall_normal_momentum_rms", np.float64),
        ],
    )
    np.save(output_dir / "diagnostics.npy", diagnostic_array)
    np.savetxt(
        output_dir / "diagnostics.csv",
        diagnostic_array,
        delimiter=",",
        header="step,div_max,div_rms,div_rel,schur_iterations,schur_abs_residual,schur_rel_residual,wall_normal_momentum_max,wall_normal_momentum_rms",
        comments="",
    )
    print(
        "Final div(u) diagnostic: "
        f"max={final_div_max:.6e}, "
        f"rms={final_div_rms:.6e}, "
        f"relative={final_div_rel:.6e}"
    )
    print(
        "Final Schur saddle solve diagnostic: "
        f"iterations={static_model.last_pressure_iterations}, "
        f"abs_residual={static_model.last_pressure_residual:.6e}, "
        f"rel_residual={static_model.last_pressure_relative_residual:.6e}"
    )
    print(
        "Final wall normal momentum diagnostic: "
        f"max={final_wall_mom_max:.6e}, "
        f"rms={final_wall_mom_rms:.6e}"
    )

metadata["completed_steps"] = final_step
metadata["elapsed_seconds"] = end - start
metadata["numerics"]["spectral_refresh"]["actual_count"] = (
    solver.integrator.refresh_count
)
(output_dir / "COMPLETE").write_text("complete\n")
write_run_metadata(output_dir, metadata, status="complete")
