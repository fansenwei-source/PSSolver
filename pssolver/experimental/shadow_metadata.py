"""Scientific metadata comparison for the Plane shadow path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pssolver.models.active_nematics import (
    Q_convention_metadata,
    positive_equilibrium_S,
)

from ._shadow_support import (
    SHADOW_CHECKPOINT_FORMAT_VERSION,
    SHADOW_RUN_SCHEMA_VERSION,
    SHADOW_SCRIPT_ID,
    require_plane_beris_edwards_runtime,
    shadow_runtime_identity_sha256,
)
from .model_execution import ExperimentalModelRuntime


def _boundary_signature(boundaries) -> list[str]:
    return [condition.kind.value for condition in boundaries.axes]


def _resolved_numerical_policy(
    runtime: ExperimentalModelRuntime,
    system_name: str,
) -> Mapping[str, object]:
    for resolved in runtime.resolved_algebraic_systems:
        if resolved.system.name != system_name:
            continue
        metadata = resolved.to_metadata()["observability"]
        policy = metadata.get("numerical_policy", {})
        if not isinstance(policy, Mapping):
            raise ValueError("resolved numerical policy is invalid")
        return policy
    raise ValueError(f"runtime is missing algebraic system {system_name!r}")


def plane_beris_edwards_shadow_signature(
    runtime: ExperimentalModelRuntime,
) -> dict[str, object]:
    """Return fields directly comparable with production Plane metadata."""

    model = require_plane_beris_edwards_runtime(runtime)
    constitutive = model.constitutive_model
    parameters = constitutive.parameters
    stokes = constitutive.stokes_system
    numerics = runtime.problem.numerics
    molecular_policy = _resolved_numerical_policy(
        runtime,
        "molecular_field",
    )
    force_policy = _resolved_numerical_policy(runtime, "nematic_force")
    return {
        "solver": {
            "shape": list(runtime.context.physical_shape),
            "spectral_shape": list(runtime.context.spectral_shape),
            "lengths": list(runtime.context.lengths),
            "dt": float(runtime.solver.dt),
            "real_dtype": str(runtime.context.real_dtype).removeprefix("torch."),
            "spectral_dtype": str(
                runtime.solver.transform_backend.spectral_dtype
            ).removeprefix("torch."),
            "transform_execution_order": (
                numerics.transform_execution_order.value
            ),
            "spectral_storage": numerics.spectral_storage.value,
        },
        "model": {
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "active_stress_prefactor": parameters.active_prefactor,
                "eta": stokes.viscosity,
                "flow_alignment_lambda": parameters.flow_alignment,
                "fric": stokes.friction,
                "ldg_A": parameters.ldg_a,
                "ldg_B": parameters.ldg_b,
                "ldg_C": parameters.ldg_c,
                "ldg_L1": parameters.ldg_l1,
                "rotational_viscosity_gamma": model.rotational_viscosity,
            },
        },
        "boundary_conditions": {
            "Q": _boundary_signature(constitutive.q_boundaries),
            "velocity_tangential": _boundary_signature(
                constitutive.tangential_boundaries
            ),
            "velocity_normal": _boundary_signature(
                constitutive.normal_boundaries
            ),
            "pressure": _boundary_signature(
                constitutive.pressure_boundaries
            ),
        },
        "numerics": {
            "dealias_rule": numerics.dealias_rule.value,
            "hermitian_axis": numerics.hermitian_axis,
            "molecular_field_linear_space": molecular_policy["linear_space"],
            "pointwise_execution": molecular_policy["pointwise_execution"],
            "projected_transform_execution": (
                numerics.projected_transform_execution.value
            ),
            "stress_divergence_sum_space": force_policy[
                "stress_divergence_sum_space"
            ],
            "spectral_refresh_interval_steps": (
                runtime.solver.integrator.spectral_refresh_interval
            ),
            "velocity_zero_mode": (
                stokes.tangential_zero_mode_policy.value
            ),
        },
    }


def plane_beris_edwards_production_signature(
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Extract the same scientific signature from production run metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("production metadata must be a mapping")
    try:
        solver = metadata["solver"]
        model = metadata["model"]
        if model["name"] != "active_nematics" or model["variant"] != (
            "beris_edwards_complete_nematic_stress_stokes"
        ):
            raise ValueError("production model identity is incompatible")
        parameters = model["parameters"]
        boundaries = metadata["boundary_conditions"]
        numerics = metadata["numerics"]
        transforms = numerics["transforms"]
        dealiasing = numerics["dealiasing"]
        pointwise = numerics["pointwise_kernels"]
        return {
            "solver": {
                "shape": list(solver["shape"]),
                "spectral_shape": list(solver["spectral_shape"]),
                "lengths": list(solver["lengths"]),
                "dt": float(solver["dt"]),
                "real_dtype": solver["real_dtype"],
                "spectral_dtype": solver["spectral_dtype"],
                "transform_execution_order": solver[
                    "transform_execution_order"
                ],
                "spectral_storage": solver["spectral_storage"],
            },
            "model": {
                "Q_convention": model["Q_convention"],
                "parameters": {
                    "active_stress_prefactor": (
                        float(parameters["alpha"])
                        * float(parameters["beta"])
                    ),
                    "eta": float(parameters["eta"]),
                    "flow_alignment_lambda": float(
                        parameters["flow_alignment_lambda"]
                    ),
                    "fric": float(parameters["fric"]),
                    "ldg_A": float(parameters["ldg_A"]),
                    "ldg_B": float(parameters["ldg_B"]),
                    "ldg_C": float(parameters["ldg_C"]),
                    "ldg_L1": float(parameters["ldg_L1"]),
                    "rotational_viscosity_gamma": float(
                        parameters["rotational_viscosity_gamma"]
                    ),
                },
            },
            "boundary_conditions": {
                "Q": list(boundaries["Q"]),
                "velocity_tangential": list(
                    boundaries["velocity_tangential"]
                ),
                "velocity_normal": list(boundaries["velocity_normal"]),
                "pressure": list(boundaries["pressure"]),
            },
            "numerics": {
                "dealias_rule": dealiasing["rule"],
                "hermitian_axis": transforms["hermitian_axis"],
                "molecular_field_linear_space": numerics[
                    "molecular_field_linear_space"
                ],
                "pointwise_execution": pointwise["effective"],
                "projected_transform_execution": transforms[
                    "projected_transform_execution"
                ],
                "stress_divergence_sum_space": numerics[
                    "stress_divergence_sum_space"
                ],
                "spectral_refresh_interval_steps": numerics[
                    "spectral_refresh"
                ]["effective_interval_steps"],
                "velocity_zero_mode": numerics["velocity_zero_mode"],
            },
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "production metadata lacks the Stage K comparison contract"
        ) from exc


def _different_paths(
    left: object,
    right: object,
    prefix: str = "",
) -> tuple[str, ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_different_paths(left[key], right[key], child))
        return tuple(paths)
    if left != right:
        return (prefix,)
    return ()


@dataclass(frozen=True, slots=True)
class ShadowMetadataComparison:
    """Auditable comparison between shadow and production configurations."""

    shadow_signature: Mapping[str, object]
    production_signature: Mapping[str, object]
    differing_paths: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.differing_paths

    def require_compatible(self) -> None:
        if self.differing_paths:
            raise ValueError(
                "shadow and production metadata differ at "
                f"{self.differing_paths!r}"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "compatible": self.compatible,
            "differing_paths": list(self.differing_paths),
            "shadow_signature": dict(self.shadow_signature),
            "production_signature": dict(self.production_signature),
        }


def compare_shadow_to_production_metadata(
    runtime: ExperimentalModelRuntime,
    production_metadata: Mapping[str, Any],
) -> ShadowMetadataComparison:
    """Compare all currently migrated trajectory-defining Plane settings."""

    shadow = plane_beris_edwards_shadow_signature(runtime)
    production = plane_beris_edwards_production_signature(
        production_metadata
    )
    return ShadowMetadataComparison(
        shadow_signature=MappingProxyType(shadow),
        production_signature=MappingProxyType(production),
        differing_paths=_different_paths(shadow, production),
    )


def build_shadow_run_metadata(
    runtime: ExperimentalModelRuntime,
    *,
    initial_condition: Mapping[str, object],
    raw_q_sha256: str,
    projected_q_sha256: str,
) -> dict[str, object]:
    """Build canonical run metadata for the opt-in shadow driver."""

    model = require_plane_beris_edwards_runtime(runtime)
    signature = plane_beris_edwards_shadow_signature(runtime)
    parameters = dict(signature["model"]["parameters"])
    try:
        equilibrium = positive_equilibrium_S(
            parameters["ldg_A"],
            parameters["ldg_B"],
            parameters["ldg_C"],
        )
    except ValueError:
        equilibrium = None
    parameters["S_bulk"] = equilibrium
    if "S_initial" in initial_condition:
        parameters["S_initial"] = initial_condition["S_initial"]
    initial_metadata = dict(initial_condition)
    initial_metadata["raw_q_sha256"] = raw_q_sha256
    initial_metadata["projected_q_sha256"] = projected_q_sha256
    return {
        "schema_version": SHADOW_RUN_SCHEMA_VERSION,
        "script": SHADOW_SCRIPT_ID,
        "solver": signature["solver"],
        "model": {
            "name": "active_nematics",
            "variant": "beris_edwards_complete_nematic_stress_stokes",
            "Q_convention": Q_convention_metadata(),
            "parameters": parameters,
        },
        "boundary_conditions": signature["boundary_conditions"],
        "numerics": signature["numerics"],
        "initial_condition": initial_metadata,
        "architecture_runtime": runtime.to_metadata(),
        "shadow_runtime_identity_sha256": (
            shadow_runtime_identity_sha256(runtime)
        ),
        "checkpointing": {
            "complete_checkpoint_format_version": (
                SHADOW_CHECKPOINT_FORMAT_VERSION
            ),
            "observation_layout": "Q/u/p production filenames",
            "transient_fields_checkpointed": False,
            "stored_algebraic_fields_checkpointed": False,
            "stored_algebraic_fields_reconstructed": True,
        },
        "completed_steps": 0,
    }


__all__ = [
    "ShadowMetadataComparison",
    "build_shadow_run_metadata",
    "compare_shadow_to_production_metadata",
    "plane_beris_edwards_production_signature",
    "plane_beris_edwards_shadow_signature",
]
