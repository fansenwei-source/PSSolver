"""Provisional Plane run-configuration components and facade decomposition.

The component graph remains disconnected from every production consumer.  A
pure one-way adapter decomposes the supported flat facade, but the facade does
not import this module.  These values do not construct a runtime, serialize
schema-v1 metadata, or select a numerical implementation.  Their local and
composition-root validation boundaries make ownership explicit before any
production consumer migrates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path

from pssolver.core import (
    DealiasRule,
    DomainSpec,
    NumericsConfig,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics.specifications import (
    BerisEdwardsMaterialRequest,
    ExtrudedDefectGasInitialConditionSpec,
)
from pssolver.presets.shendruk import (
    ShendrukPlaneParameterRequest,
    ShendrukPlanePreset,
    resolve_shendruk_plane_preset,
)
from pssolver.systems.stokes import (
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)

from .plane_beris_edwards import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PLANE_HERMITIAN_AXIS,
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
)


def _positive_finite(value: object, description: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


def _positive_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{description} must be a positive integer")
    return value


def _non_negative_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def _require_bool(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{description} must be a bool")
    return value


def _path(value: object, description: str) -> Path:
    try:
        return Path(value)
    except TypeError as exc:
        raise TypeError(f"{description} must be path-like") from exc


def _optional_path(value: object, description: str) -> Path | None:
    if value is None:
        return None
    return _path(value, description)


def _require_choice(
    value: object,
    choices: frozenset[str],
    description: str,
) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{description} must be one of {tuple(sorted(choices))!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class PlaneBerisEdwardsPhysicsSpec:
    """Plane-specific composition of tensor-free physical requests."""

    material: BerisEdwardsMaterialRequest
    shendruk_request: ShendrukPlaneParameterRequest
    stokes: IncompressibleStokesSystemSpec
    requested_friction_mode_fric: float

    def __post_init__(self) -> None:
        declarations = (
            ("material", self.material, BerisEdwardsMaterialRequest),
            (
                "shendruk_request",
                self.shendruk_request,
                ShendrukPlaneParameterRequest,
            ),
            ("stokes", self.stokes, IncompressibleStokesSystemSpec),
        )
        for name, value, value_type in declarations:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be a {value_type.__name__}")

        requested_friction = self.requested_friction_mode_fric
        if (
            not isinstance(requested_friction, (int, float))
            or isinstance(requested_friction, bool)
            or not math.isfinite(float(requested_friction))
        ):
            raise ValueError(
                "requested_friction_mode_fric must be a finite int or float"
            )

        policy = self.stokes.tangential_zero_mode_policy
        if policy not in (
            TangentialZeroModePolicy.ZERO_MEAN,
            TangentialZeroModePolicy.FRICTION,
        ):
            raise ValueError(
                "Plane Stokes zero-mode policy must be zero_mean or friction"
            )
        if policy is TangentialZeroModePolicy.FRICTION:
            if requested_friction <= 0.0:
                raise ValueError(
                    "requested_friction_mode_fric must be positive in "
                    "friction mode"
                )
            if self.stokes.friction != requested_friction:
                raise ValueError(
                    "effective Stokes friction must equal the requested "
                    "friction-mode value"
                )

    def to_metadata(self) -> dict[str, object]:
        """Return provisional component metadata, not schema-v1 metadata."""

        return {
            "material": self.material.to_metadata(),
            "shendruk_request": self.shendruk_request.to_metadata(),
            "stokes": self.stokes.to_metadata(),
            "requested_friction_mode_fric": (
                self.requested_friction_mode_fric
            ),
        }


@dataclass(frozen=True, slots=True)
class PlaneTimeSteppingSpec:
    """Time increment and resolved dynamic-spectrum refresh schedule."""

    dt: float
    spectral_refresh: SpectralRefreshSpec

    def __post_init__(self) -> None:
        dt = _positive_finite(self.dt, "dt")
        if not isinstance(self.spectral_refresh, SpectralRefreshSpec):
            raise TypeError(
                "spectral_refresh must be a SpectralRefreshSpec"
            )
        # Consistency between dt and the resolved refresh interval is a
        # composition-root invariant.  Keeping it out of this disconnected
        # leaf avoids moving legacy direct-constructor validation in P2.1.
        object.__setattr__(self, "dt", dt)

    def to_metadata(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "spectral_refresh": self.spectral_refresh.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class PlaneBerisEdwardsExecutionSpec:
    """Requested implementation and device policies for one Plane run."""

    device: str
    tf32: str
    molecular_field_linear_space: str
    stress_divergence_sum_space: str
    pointwise_execution: str
    disable_q_gradient_reuse: bool
    runtime_path: PlaneRuntimePath

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        _require_choice(self.tf32, frozenset({"off", "on"}), "tf32")
        _require_choice(
            self.molecular_field_linear_space,
            frozenset({"physical", "spectral"}),
            "molecular_field_linear_space",
        )
        _require_choice(
            self.stress_divergence_sum_space,
            frozenset({"physical", "spectral"}),
            "stress_divergence_sum_space",
        )
        _require_choice(
            self.pointwise_execution,
            frozenset({"eager", "compile"}),
            "pointwise_execution",
        )
        _require_bool(
            self.disable_q_gradient_reuse,
            "disable_q_gradient_reuse",
        )
        if not isinstance(self.runtime_path, PlaneRuntimePath):
            raise TypeError("runtime_path must be a PlaneRuntimePath")
        # The separated-canary/cache compatibility rule also spans facade
        # ownership groups and remains a later composition-root check.

    def to_metadata(self) -> dict[str, object]:
        return {
            "device": self.device,
            "tf32": self.tf32,
            "molecular_field_linear_space": (
                self.molecular_field_linear_space
            ),
            "stress_divergence_sum_space": (
                self.stress_divergence_sum_space
            ),
            "pointwise_execution": self.pointwise_execution,
            "disable_q_gradient_reuse": self.disable_q_gradient_reuse,
            "runtime_path": self.runtime_path.value,
        }


@dataclass(frozen=True, slots=True)
class PlaneWorkflowSpec:
    """Output, observation, checkpoint, and restart choices."""

    output_dir: Path
    steps: int
    save_start_step: int
    save_interval: int
    diagnostic_interval: int
    diagnostics: bool
    save_hydrodynamics: bool
    checkpoint_interval: int | None
    restart_from: Path | None

    def __post_init__(self) -> None:
        output_dir = _path(self.output_dir, "output_dir")
        steps = _positive_integer(self.steps, "steps")
        save_start = _non_negative_integer(
            self.save_start_step,
            "save_start_step",
        )
        if save_start > steps:
            raise ValueError("save_start_step must not exceed steps")
        _positive_integer(self.save_interval, "save_interval")
        _positive_integer(
            self.diagnostic_interval,
            "diagnostic_interval",
        )
        _require_bool(self.diagnostics, "diagnostics")
        _require_bool(self.save_hydrodynamics, "save_hydrodynamics")
        checkpoint_interval = self.checkpoint_interval
        if checkpoint_interval is not None:
            _positive_integer(checkpoint_interval, "checkpoint_interval")
        restart_from = _optional_path(self.restart_from, "restart_from")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "save_start_step", save_start)
        object.__setattr__(self, "restart_from", restart_from)

    def to_metadata(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "steps": self.steps,
            "save_start_step": self.save_start_step,
            "save_interval": self.save_interval,
            "diagnostic_interval": self.diagnostic_interval,
            "diagnostics": self.diagnostics,
            "save_hydrodynamics": self.save_hydrodynamics,
            "checkpoint_interval": self.checkpoint_interval,
            "restart_from": (
                str(self.restart_from)
                if self.restart_from is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class PlaneInvocationSpec:
    """Non-scientific invocation and validation provenance controls."""

    validation_config_sha256: str | None
    dry_run: bool

    def __post_init__(self) -> None:
        digest = self.validation_config_sha256
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "validation_config_sha256 must be 64 lowercase hexadecimal "
                "characters or None"
            )
        _require_bool(self.dry_run, "dry_run")

    def to_metadata(self) -> dict[str, object]:
        return {
            "validation_config_sha256": self.validation_config_sha256,
            "dry_run": self.dry_run,
        }


def _validate_refresh_consistency(value: PlaneTimeSteppingSpec) -> None:
    refresh = value.spectral_refresh
    interval_fields = (
        refresh.requested_interval_time,
        refresh.requested_interval_steps,
        refresh.effective_interval_steps,
        refresh.effective_interval_time,
    )
    if refresh.mode == "disabled":
        if any(item is not None for item in interval_fields):
            raise ValueError(
                "disabled spectral refresh must not define intervals"
            )
        return

    effective_steps = refresh.effective_interval_steps
    if (
        not isinstance(effective_steps, int)
        or isinstance(effective_steps, bool)
        or effective_steps <= 0
    ):
        raise ValueError(
            "enabled spectral refresh requires positive effective steps"
        )
    expected_time = effective_steps * value.dt
    effective_time = refresh.effective_interval_time
    if (
        not isinstance(effective_time, Real)
        or isinstance(effective_time, bool)
        or not math.isfinite(float(effective_time))
        or not math.isclose(
            float(effective_time),
            expected_time,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "effective spectral refresh time must equal steps * dt"
        )

    if refresh.mode == "steps":
        if refresh.requested_interval_time is not None:
            raise ValueError(
                "step-based spectral refresh must not request physical time"
            )
        requested_steps = refresh.requested_interval_steps
        if (
            not isinstance(requested_steps, int)
            or isinstance(requested_steps, bool)
            or requested_steps <= 0
            or requested_steps != effective_steps
        ):
            raise ValueError(
                "step-based spectral refresh must preserve requested steps"
            )
        return

    requested_time = refresh.requested_interval_time
    if refresh.requested_interval_steps is not None:
        raise ValueError(
            "physical-time spectral refresh must not request step count"
        )
    if (
        not isinstance(requested_time, Real)
        or isinstance(requested_time, bool)
        or not math.isfinite(float(requested_time))
        or float(requested_time) <= 0.0
        or not math.isclose(
            float(requested_time),
            expected_time,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "physical-time spectral refresh must be an integer multiple of dt"
        )


@dataclass(frozen=True, slots=True)
class PlaneBerisEdwardsRunComponents:
    """Disconnected, internally coherent component graph for one Plane run."""

    geometry: PlaneSlab
    boundaries: PlaneFreeSlipBoundaryConditions
    effective_boundaries: PlaneFreeSlipBoundaryConditions
    numerics: NumericsConfig
    physics: PlaneBerisEdwardsPhysicsSpec
    preset: ShendrukPlanePreset
    time_stepping: PlaneTimeSteppingSpec
    initial_condition: ExtrudedDefectGasInitialConditionSpec
    execution: PlaneBerisEdwardsExecutionSpec
    workflow: PlaneWorkflowSpec
    invocation: PlaneInvocationSpec

    def __post_init__(self) -> None:
        components = (
            ("geometry", self.geometry, PlaneSlab),
            (
                "boundaries",
                self.boundaries,
                PlaneFreeSlipBoundaryConditions,
            ),
            (
                "effective_boundaries",
                self.effective_boundaries,
                PlaneFreeSlipBoundaryConditions,
            ),
            ("numerics", self.numerics, NumericsConfig),
            ("physics", self.physics, PlaneBerisEdwardsPhysicsSpec),
            ("preset", self.preset, ShendrukPlanePreset),
            (
                "time_stepping",
                self.time_stepping,
                PlaneTimeSteppingSpec,
            ),
            (
                "initial_condition",
                self.initial_condition,
                ExtrudedDefectGasInitialConditionSpec,
            ),
            (
                "execution",
                self.execution,
                PlaneBerisEdwardsExecutionSpec,
            ),
            ("workflow", self.workflow, PlaneWorkflowSpec),
            ("invocation", self.invocation, PlaneInvocationSpec),
        )
        for name, value, value_type in components:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be a {value_type.__name__}")

        if (
            self.geometry.domain.ndim != 3
            or self.geometry.periodic_axes != (0, 1)
            or self.geometry.bounded_axes != (2,)
        ):
            raise ValueError(
                "geometry must be a three-dimensional Plane slab with "
                "periodic axes (0, 1) and bounded axis 2"
            )
        if self.effective_boundaries != PLANE_FREE_SLIP_BOUNDARIES:
            raise ValueError(
                "effective_boundaries must be the qualified Plane free-slip "
                "boundary declaration"
            )

        request = self.physics.shendruk_request
        material = self.physics.material
        expected_preset = resolve_shendruk_plane_preset(
            activity_number=request.activity_number,
            height=self.geometry.domain.lengths[2],
            parameterization=request.parameterization,
            frank_k=request.frank_k,
            coefficient_min=request.coefficient_min,
            coefficient_max=request.coefficient_max,
            ldg_a=material.ldg_a,
            ldg_b=material.ldg_b,
            ldg_c=material.ldg_c,
            gamma=material.gamma,
        )
        if self.preset != expected_preset:
            raise ValueError(
                "preset must match the geometry, material, and raw Shendruk "
                "request"
            )

        nz = self.geometry.domain.shape[2]
        if any(mode >= nz for mode in self.initial_condition.twist_modes):
            raise ValueError(
                "initial-condition twist modes must satisfy 1 <= mode < nz"
            )
        _validate_refresh_consistency(self.time_stepping)

        if self.numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF:
            if (
                self.numerics.hermitian_axis != PLANE_HERMITIAN_AXIS
                or self.numerics.hermitian_axis
                not in self.geometry.periodic_axes
            ):
                raise ValueError(
                    "Hermitian-half storage must use the qualified periodic "
                    "Plane axis"
                )
        if (
            self.execution.runtime_path is PlaneRuntimePath.SEPARATED_CANARY
            and self.execution.disable_q_gradient_reuse
        ):
            raise ValueError(
                "separated_canary does not accept legacy Q-gradient cache "
                "flags"
            )

    def to_metadata(self) -> dict[str, object]:
        """Return provisional nested metadata, never schema-v1 authority."""

        return {
            "geometry": self.geometry.to_metadata(),
            "boundaries": self.boundaries.to_metadata(),
            "effective_boundaries": self.effective_boundaries.to_metadata(),
            "numerics": self.numerics.to_metadata(),
            "physics": self.physics.to_metadata(),
            "preset": self.preset.to_metadata(),
            "time_stepping": self.time_stepping.to_metadata(),
            "initial_condition": self.initial_condition.to_metadata(),
            "execution": self.execution.to_metadata(),
            "workflow": self.workflow.to_metadata(),
            "invocation": self.invocation.to_metadata(),
        }


def decompose_plane_beris_edwards_run_spec(
    legacy_spec: PlaneBerisEdwardsRunSpec,
) -> PlaneBerisEdwardsRunComponents:
    """Purely decompose one qualified flat facade into provisional parts.

    This adapter deliberately reads flat fields instead of the facade's
    derived ``geometry``, ``numerics``, or ``shendruk_preset`` properties.
    That keeps a later facade-delegation phase from introducing recursion.
    Schema-v1 serialization and identity remain owned by ``legacy_spec``.
    """

    if not isinstance(legacy_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError(
            "legacy_spec must be a PlaneBerisEdwardsRunSpec"
        )

    geometry = PlaneSlab(
        DomainSpec(
            (legacy_spec.nx, legacy_spec.ny, legacy_spec.nz),
            (legacy_spec.lx, legacy_spec.ly, legacy_spec.height),
        ),
        wall_normal_axis=2,
    )
    numerics = NumericsConfig(
        precision=Precision(legacy_spec.dtype),
        dealias_rule=DealiasRule(legacy_spec.dealias_rule),
        transform_execution_order=TransformExecutionOrder(
            legacy_spec.transform_execution_order
        ),
        projected_transform_execution=ProjectedTransformExecution(
            legacy_spec.projected_transform_execution
        ),
        spectral_storage=SpectralStorage(legacy_spec.spectral_storage),
        hermitian_axis=(
            PLANE_HERMITIAN_AXIS
            if legacy_spec.spectral_storage == "hermitian_half"
            else None
        ),
    )
    material = BerisEdwardsMaterialRequest(
        ldg_a=legacy_spec.ldg_a,
        ldg_b=legacy_spec.ldg_b,
        ldg_c=legacy_spec.ldg_c,
        gamma=legacy_spec.gamma,
        flow_alignment=legacy_spec.flow_alignment,
        beta=legacy_spec.beta,
    )
    shendruk_request = ShendrukPlaneParameterRequest(
        activity_number=legacy_spec.activity_number,
        parameterization=legacy_spec.parameterization,
        frank_k=legacy_spec.frank_k,
        coefficient_min=legacy_spec.coefficient_min,
        coefficient_max=legacy_spec.coefficient_max,
    )
    preset = resolve_shendruk_plane_preset(
        activity_number=legacy_spec.activity_number,
        height=legacy_spec.height,
        parameterization=legacy_spec.parameterization,
        frank_k=legacy_spec.frank_k,
        coefficient_min=legacy_spec.coefficient_min,
        coefficient_max=legacy_spec.coefficient_max,
        ldg_a=legacy_spec.ldg_a,
        ldg_b=legacy_spec.ldg_b,
        ldg_c=legacy_spec.ldg_c,
        gamma=legacy_spec.gamma,
    )
    zero_mode_policy = TangentialZeroModePolicy(
        legacy_spec.zero_mode_policy
    )
    effective_friction = (
        legacy_spec.friction_mode_fric
        if zero_mode_policy is TangentialZeroModePolicy.FRICTION
        else 0.0
    )
    stokes = IncompressibleStokesSystemSpec(
        name="flow",
        force_components=("force_x", "force_y", "force_z"),
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=legacy_spec.eta,
        friction=effective_friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=zero_mode_policy,
    )

    return PlaneBerisEdwardsRunComponents(
        geometry=geometry,
        boundaries=legacy_spec.boundaries,
        effective_boundaries=PLANE_FREE_SLIP_BOUNDARIES,
        numerics=numerics,
        physics=PlaneBerisEdwardsPhysicsSpec(
            material=material,
            shendruk_request=shendruk_request,
            stokes=stokes,
            requested_friction_mode_fric=(
                legacy_spec.friction_mode_fric
            ),
        ),
        preset=preset,
        time_stepping=PlaneTimeSteppingSpec(
            dt=legacy_spec.dt,
            spectral_refresh=legacy_spec.spectral_refresh,
        ),
        initial_condition=ExtrudedDefectGasInitialConditionSpec(
            seed=legacy_spec.seed,
            num_defect_pairs=legacy_spec.num_defect_pairs,
            defect_min_separation=legacy_spec.defect_min_separation,
            defect_core_radius=legacy_spec.defect_core_radius,
            background_angle=legacy_spec.background_angle,
            twist_amplitude=legacy_spec.twist_amplitude,
            twist_modes=legacy_spec.twist_modes,
            initial_s=legacy_spec.initial_s,
        ),
        execution=PlaneBerisEdwardsExecutionSpec(
            device=legacy_spec.device,
            tf32=legacy_spec.tf32,
            molecular_field_linear_space=(
                legacy_spec.molecular_field_linear_space
            ),
            stress_divergence_sum_space=(
                legacy_spec.stress_divergence_sum_space
            ),
            pointwise_execution=legacy_spec.pointwise_execution,
            disable_q_gradient_reuse=(
                legacy_spec.disable_q_gradient_reuse
            ),
            runtime_path=legacy_spec.runtime_path,
        ),
        workflow=PlaneWorkflowSpec(
            output_dir=legacy_spec.output_dir,
            steps=legacy_spec.steps,
            save_start_step=legacy_spec.save_start_step,
            save_interval=legacy_spec.save_interval,
            diagnostic_interval=legacy_spec.diagnostic_interval,
            diagnostics=legacy_spec.diagnostics,
            save_hydrodynamics=legacy_spec.save_hydrodynamics,
            checkpoint_interval=legacy_spec.checkpoint_interval,
            restart_from=legacy_spec.restart_from,
        ),
        invocation=PlaneInvocationSpec(
            validation_config_sha256=(
                legacy_spec.validation_config_sha256
            ),
            dry_run=legacy_spec.dry_run,
        ),
    )


__all__ = [
    "decompose_plane_beris_edwards_run_spec",
    "PlaneBerisEdwardsPhysicsSpec",
    "PlaneBerisEdwardsRunComponents",
    "PlaneBerisEdwardsExecutionSpec",
    "PlaneInvocationSpec",
    "PlaneTimeSteppingSpec",
    "PlaneWorkflowSpec",
]
