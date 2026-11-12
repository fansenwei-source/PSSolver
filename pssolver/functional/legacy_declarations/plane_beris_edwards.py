import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pssolver.core import DomainSpec, NumericsConfig
from pssolver.core.device_tokens import validate_execution_device
from pssolver.core.numerics import (
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
    SPECTRAL_STORAGE_MODES,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards_builders import (
    build_plane_beris_edwards_domain,
    build_plane_beris_edwards_geometry,
    build_plane_beris_edwards_numerics,
    build_plane_beris_edwards_shendruk_preset,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards_declarations import (
    DEFAULT_FRICTION_MODE_FRIC,
    DEFAULT_PLANE_RUNTIME_PATH,
    DEFAULT_SPECTRAL_REFRESH_TIME,
    DEFAULT_ZERO_MODE_POLICY,
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards_schema_v1 import (
    PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER,
    PLANE_RUN_SPEC_SCHEMA_VERSION,
)
from pssolver.geometries import PlaneSlab
from pssolver.geometries.plane_numerics import DEFAULT_PLANE_SPECTRAL_STORAGE, PLANE_HERMITIAN_AXIS
from pssolver.models.active_nematics.beris_edwards import (
    DEFAULT_POINTWISE_EXECUTION,
    POINTWISE_EXECUTION_MODES,
)
from pssolver.models.active_nematics.stokes import (
    DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE,
    DEFAULT_STRESS_DIVERGENCE_SUM_SPACE,
)
from pssolver.presets import ShendrukPlanePreset


PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES = (
    "pssolver/applications/__init__.py",
    "pssolver/solver.py",
    "pssolver/Field.py",
    "pssolver/PDEmodel.py",
    "pssolver/integrator.py",
    "pssolver/plane.py",
    "pssolver/transforms.py",
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
    "pssolver/adapters/legacy_boundaries.py",
    "pssolver/configuration/__init__.py",
    "pssolver/configuration/plane_beris_edwards_builders.py",
    "pssolver/configuration/plane_beris_edwards_component_graph.py",
    "pssolver/configuration/plane_beris_edwards_components.py",
    "pssolver/configuration/plane_beris_edwards_declarations.py",
    "pssolver/configuration/plane_beris_edwards_schema_v1.py",
    "pssolver/configuration/plane_beris_edwards.py",
    "pssolver/runtime/__init__.py",
    "pssolver/workflows/__init__.py",
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
    "pssolver/models/active_nematics/specifications.py",
    "pssolver/models/active_nematics/stokes.py",
    "pssolver/models/active_nematics/initial_conditions.py",
    "pssolver/presets/__init__.py",
    "pssolver/presets/shendruk.py",
    "pssolver/systems/__init__.py",
    "pssolver/systems/algebraic.py",
    "pssolver/systems/stokes.py",
    "pssolver/configuration/legacy_plane.py",
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
    "pssolver/runtime/legacy_entry.py",
    "pssolver/runtime/plane_observations.py",
    "pssolver/api/runner.py",
    "pssolver/configuration/application_registry.py",
    "pssolver/configuration/package_construction.py",
    "pssolver/experimental/boundary_pilots.py",
)


@dataclass(frozen=True, slots=True)
class PlaneBerisEdwardsRunSpec:
    """Complete tensor-free configuration for one production Plane run."""

    activity_number: float
    output_dir: Path
    height: float
    parameterization: str
    frank_k: float
    coefficient_min: float
    coefficient_max: float
    lx: float
    ly: float
    nx: int
    ny: int
    nz: int
    dt: float
    steps: int
    save_start_step: int
    save_interval: int
    diagnostic_interval: int
    seed: int
    num_defect_pairs: int
    defect_min_separation: float
    defect_core_radius: float
    background_angle: float
    twist_amplitude: float
    twist_modes: tuple[int, ...]
    ldg_a: float
    ldg_b: float
    ldg_c: float
    gamma: float
    flow_alignment: float
    eta: float
    zero_mode_policy: str
    friction_mode_fric: float
    dealias_rule: str
    projected_transform_execution: str
    beta: float
    initial_s: float
    device: str
    dtype: str
    molecular_field_linear_space: str
    stress_divergence_sum_space: str
    pointwise_execution: str
    transform_execution_order: str
    spectral_storage: str
    tf32: str
    spectral_refresh: SpectralRefreshSpec
    diagnostics: bool
    disable_q_gradient_reuse: bool
    save_hydrodynamics: bool
    validation_config_sha256: str | None
    dry_run: bool
    checkpoint_interval: int | None = None
    restart_from: Path | None = None
    runtime_path: PlaneRuntimePath = PlaneRuntimePath.LEGACY_PRODUCTION
    boundaries: PlaneFreeSlipBoundaryConditions = PLANE_FREE_SLIP_BOUNDARIES

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.restart_from is not None:
            object.__setattr__(self, "restart_from", Path(self.restart_from))
        object.__setattr__(self, "twist_modes", tuple(self.twist_modes))
        object.__setattr__(
            self,
            "device",
            validate_execution_device(self.device),
        )
        if not isinstance(self.spectral_refresh, SpectralRefreshSpec):
            raise TypeError("spectral_refresh must be a SpectralRefreshSpec")
        if not isinstance(self.boundaries, PlaneFreeSlipBoundaryConditions):
            raise TypeError(
                "boundaries must be PlaneFreeSlipBoundaryConditions"
            )
        if not isinstance(self.runtime_path, PlaneRuntimePath):
            raise TypeError("runtime_path must be a PlaneRuntimePath")

    @property
    def S_initial(self) -> float:
        """Compatibility spelling used by the current production driver."""

        return self.initial_s

    @property
    def spectral_refresh_mode(self) -> str:
        return self.spectral_refresh.mode

    @property
    def spectral_refresh_requested_time(self) -> float | None:
        return self.spectral_refresh.requested_interval_time

    @property
    def spectral_refresh_requested_steps(self) -> int | None:
        return self.spectral_refresh.requested_interval_steps

    @property
    def spectral_refresh_interval_steps(self) -> int | None:
        return self.spectral_refresh.effective_interval_steps

    @property
    def spectral_refresh_effective_time(self) -> float | None:
        return self.spectral_refresh.effective_interval_time

    @property
    def shendruk_preset(self) -> ShendrukPlanePreset:
        return build_plane_beris_edwards_shendruk_preset(
            activity_number=self.activity_number,
            height=self.height,
            parameterization=self.parameterization,
            frank_k=self.frank_k,
            coefficient_min=self.coefficient_min,
            coefficient_max=self.coefficient_max,
            ldg_a=self.ldg_a,
            ldg_b=self.ldg_b,
            ldg_c=self.ldg_c,
            gamma=self.gamma,
        )

    @property
    def domain(self) -> DomainSpec:
        return build_plane_beris_edwards_domain(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            lx=self.lx,
            ly=self.ly,
            height=self.height,
        )

    @property
    def geometry(self) -> PlaneSlab:
        return build_plane_beris_edwards_geometry(self.domain)

    @property
    def numerics(self) -> NumericsConfig:
        return build_plane_beris_edwards_numerics(
            dtype=self.dtype,
            dealias_rule=self.dealias_rule,
            transform_execution_order=self.transform_execution_order,
            projected_transform_execution=(
                self.projected_transform_execution
            ),
            spectral_storage=self.spectral_storage,
            hermitian_axis=PLANE_HERMITIAN_AXIS,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return the complete resolved, JSON-compatible run specification."""

        return PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER.to_metadata(self)

    def canonical_sha256(self) -> str:
        return PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER.canonical_sha256(
            self
        )

    def runtime_identity_metadata(self) -> dict[str, object]:
        """Return the numerical identity required for same-backend restart."""

        return (
            PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER
            .runtime_identity_metadata(self)
        )

    def runtime_identity_sha256(self) -> str:
        return (
            PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER
            .runtime_identity_sha256(self)
        )

    def identity_metadata(self) -> dict[str, object]:
        return PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER.identity_metadata(
            self
        )

    def runtime_selection_metadata(self) -> dict[str, object]:
        """Return additive requested/effective runtime-path metadata."""

        return (
            PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER
            .runtime_selection_metadata(self)
        )


def _resolve_spectral_refresh(
    *,
    dt: float,
    spectral_refresh_time: float | None,
    spectral_refresh_steps: int | None,
    disable_spectral_refresh: bool,
) -> SpectralRefreshSpec:
    if spectral_refresh_steps is not None:
        if spectral_refresh_steps <= 0:
            raise ValueError("--spectral-refresh-steps must be positive")
        return SpectralRefreshSpec(
            mode="steps",
            requested_interval_time=None,
            requested_interval_steps=spectral_refresh_steps,
            effective_interval_steps=spectral_refresh_steps,
            effective_interval_time=spectral_refresh_steps * dt,
        )
    if disable_spectral_refresh:
        return SpectralRefreshSpec(
            mode="disabled",
            requested_interval_time=None,
            requested_interval_steps=None,
            effective_interval_steps=None,
            effective_interval_time=None,
        )
    requested_time = (
        DEFAULT_SPECTRAL_REFRESH_TIME
        if spectral_refresh_time is None
        else spectral_refresh_time
    )
    if not math.isfinite(requested_time) or requested_time <= 0:
        raise ValueError("--spectral-refresh-time must be positive and finite")
    interval_ratio = requested_time / dt
    interval_steps = int(round(interval_ratio))
    if interval_steps <= 0 or not math.isclose(
        interval_ratio,
        interval_steps,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "--spectral-refresh-time must be an integer multiple of --dt; "
            f"got time/dt={interval_ratio:.17g}"
        )
    return SpectralRefreshSpec(
        mode="physical_time",
        requested_interval_time=requested_time,
        requested_interval_steps=None,
        effective_interval_steps=interval_steps,
        effective_interval_time=interval_steps * dt,
    )


def create_plane_beris_edwards_run_spec(
    *,
    activity_number: float,
    output_dir: str | Path,
    height: float = 20.0,
    parameterization: str = "paper-window",
    frank_k: float = 0.01,
    coefficient_min: float = 0.01,
    coefficient_max: float = 0.05,
    lx: float = 100.0,
    ly: float = 100.0,
    nx: int = 256,
    ny: int = 256,
    nz: int = 64,
    dt: float = 1.0e-2,
    steps: int = 10_000,
    save_start_step: int = 5_000,
    save_interval: int = 500,
    diagnostic_interval: int = 100,
    seed: int = 24,
    num_defect_pairs: int = 6,
    defect_min_separation: float = 10.0,
    defect_core_radius: float = 1.5,
    background_angle: float = 0.0,
    twist_amplitude: float = 0.01,
    twist_modes: Sequence[int] = (1, 2, 3),
    ldg_a: float = 0.0,
    ldg_b: float = -0.3,
    ldg_c: float = 0.3,
    gamma: float = 2.94,
    flow_alignment: float = 0.3,
    eta: float = 2.0 / 3.0,
    zero_mode_policy: str = DEFAULT_ZERO_MODE_POLICY,
    friction_mode_fric: float = DEFAULT_FRICTION_MODE_FRIC,
    dealias_rule: str = DEFAULT_DEALIAS_RULE,
    projected_transform_execution: str = (
        DEFAULT_PROJECTED_TRANSFORM_EXECUTION
    ),
    beta: float = -1.0,
    initial_s: float = 1.0 / 3.0,
    device: str = "auto",
    dtype: str = "float32",
    molecular_field_linear_space: str = (
        DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE
    ),
    stress_divergence_sum_space: str = (
        DEFAULT_STRESS_DIVERGENCE_SUM_SPACE
    ),
    pointwise_execution: str = DEFAULT_POINTWISE_EXECUTION,
    transform_execution_order: str = DEFAULT_TRANSFORM_EXECUTION_ORDER,
    spectral_storage: str = DEFAULT_PLANE_SPECTRAL_STORAGE,
    tf32: str = "off",
    spectral_refresh_time: float | None = None,
    spectral_refresh_steps: int | None = None,
    disable_spectral_refresh: bool = False,
    diagnostics: bool = False,
    disable_q_gradient_reuse: bool = False,
    save_hydrodynamics: bool = False,
    validation_config_sha256: str | None = None,
    dry_run: bool = False,
    checkpoint_interval: int | None = None,
    restart_from: str | Path | None = None,
    runtime_path: str | PlaneRuntimePath = DEFAULT_PLANE_RUNTIME_PATH,
) -> PlaneBerisEdwardsRunSpec:
    """Create and validate the canonical programmatic run specification."""

    twist_modes = tuple(twist_modes)
    positive_values = {
        "activity_number": activity_number,
        "height": height,
        "frank_k": frank_k,
        "coefficient_min": coefficient_min,
        "coefficient_max": coefficient_max,
        "lx": lx,
        "ly": ly,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dt": dt,
        "steps": steps,
        "save_interval": save_interval,
        "diagnostic_interval": diagnostic_interval,
        "gamma": gamma,
    }
    invalid = {
        name: value for name, value in positive_values.items() if value <= 0
    }
    if invalid:
        raise ValueError(f"these values must be positive: {invalid}")
    if initial_s <= 0:
        raise ValueError("--initial-s must be positive")
    if num_defect_pairs <= 0:
        raise ValueError("--num-defect-pairs must be positive")
    if defect_min_separation <= 0:
        raise ValueError("--defect-min-separation must be positive")
    if defect_core_radius <= 0:
        raise ValueError("--defect-core-radius must be positive")
    if twist_amplitude < 0:
        raise ValueError("--twist-amplitude must be non-negative")
    if len(set(twist_modes)) != len(twist_modes):
        raise ValueError("--twist-modes must be unique")
    if any(mode <= 0 or mode >= nz for mode in twist_modes):
        raise ValueError("--twist-modes must satisfy 1 <= mode < nz")
    if zero_mode_policy not in {"zero_mean", "friction"}:
        raise ValueError("invalid zero-mode policy")
    if zero_mode_policy == "friction" and friction_mode_fric <= 0:
        raise ValueError(
            "--friction-mode-fric must be positive in friction mode"
        )
    if coefficient_min >= coefficient_max:
        raise ValueError(
            "--coefficient-min must be smaller than --coefficient-max"
        )
    try:
        resolved_runtime_path = PlaneRuntimePath(runtime_path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid runtime path: {runtime_path!r}") from exc
    if resolved_runtime_path is PlaneRuntimePath.SEPARATED_CANARY:
        raise ValueError("separated_canary is retired; use v0.4.x (maint/v0.4)")
    if checkpoint_interval is not None and (
        not isinstance(checkpoint_interval, int)
        or isinstance(checkpoint_interval, bool)
        or checkpoint_interval <= 0
    ):
        raise ValueError("--checkpoint-interval must be positive")
    supported_choices = (
        (parameterization, {"paper-window", "fixed-k"}, "parameterization"),
        (dealias_rule, set(DEALIAS_RULE_FRACTIONS), "dealias rule"),
        (
            projected_transform_execution,
            set(PROJECTED_TRANSFORM_EXECUTION_MODES),
            "projected transform execution",
        ),
        (dtype, {"float32", "float64"}, "dtype"),
        (
            molecular_field_linear_space,
            {"physical", "spectral"},
            "molecular-field linear space",
        ),
        (
            stress_divergence_sum_space,
            {"physical", "spectral"},
            "stress-divergence sum space",
        ),
        (
            pointwise_execution,
            set(POINTWISE_EXECUTION_MODES),
            "pointwise execution",
        ),
        (
            transform_execution_order,
            {"legacy", "real_first"},
            "transform execution order",
        ),
        (spectral_storage, set(SPECTRAL_STORAGE_MODES), "spectral storage"),
        (tf32, {"off", "on"}, "TF32 policy"),
    )
    for value, choices, description in supported_choices:
        if value not in choices:
            raise ValueError(f"invalid {description}: {value!r}")
    if (
        projected_transform_execution == "truncated"
        and dealias_rule == "none"
    ):
        raise ValueError(
            "--projected-transform-execution truncated requires enabled "
            "dealiasing"
        )
    if save_start_step < 0 or (
        restart_from is None and save_start_step > steps
    ):
        raise ValueError(
            "--save-start-step must lie between 0 and --steps for a fresh run"
        )
    if validation_config_sha256 is not None and (
        len(validation_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in validation_config_sha256
        )
    ):
        raise ValueError(
            "--validation-config-sha256 must be exactly 64 lowercase "
            "hexadecimal characters"
        )
    if (
        spectral_storage == "hermitian_half"
        and transform_execution_order != "real_first"
    ):
        raise ValueError(
            "--spectral-storage hermitian_half requires "
            "--transform-execution-order real_first"
        )
    refresh = _resolve_spectral_refresh(
        dt=dt,
        spectral_refresh_time=spectral_refresh_time,
        spectral_refresh_steps=spectral_refresh_steps,
        disable_spectral_refresh=disable_spectral_refresh,
    )
    spec = PlaneBerisEdwardsRunSpec(
        activity_number=activity_number,
        output_dir=Path(output_dir),
        height=height,
        parameterization=parameterization,
        frank_k=frank_k,
        coefficient_min=coefficient_min,
        coefficient_max=coefficient_max,
        lx=lx,
        ly=ly,
        nx=nx,
        ny=ny,
        nz=nz,
        dt=dt,
        steps=steps,
        save_start_step=save_start_step,
        save_interval=save_interval,
        diagnostic_interval=diagnostic_interval,
        seed=seed,
        num_defect_pairs=num_defect_pairs,
        defect_min_separation=defect_min_separation,
        defect_core_radius=defect_core_radius,
        background_angle=background_angle,
        twist_amplitude=twist_amplitude,
        twist_modes=twist_modes,
        ldg_a=ldg_a,
        ldg_b=ldg_b,
        ldg_c=ldg_c,
        gamma=gamma,
        flow_alignment=flow_alignment,
        eta=eta,
        zero_mode_policy=zero_mode_policy,
        friction_mode_fric=friction_mode_fric,
        dealias_rule=dealias_rule,
        projected_transform_execution=projected_transform_execution,
        beta=beta,
        initial_s=initial_s,
        device=device,
        dtype=dtype,
        molecular_field_linear_space=molecular_field_linear_space,
        stress_divergence_sum_space=stress_divergence_sum_space,
        pointwise_execution=pointwise_execution,
        transform_execution_order=transform_execution_order,
        spectral_storage=spectral_storage,
        tf32=tf32,
        spectral_refresh=refresh,
        diagnostics=diagnostics,
        disable_q_gradient_reuse=disable_q_gradient_reuse,
        save_hydrodynamics=save_hydrodynamics,
        validation_config_sha256=validation_config_sha256,
        dry_run=dry_run,
        checkpoint_interval=checkpoint_interval,
        restart_from=(Path(restart_from) if restart_from is not None else None),
        runtime_path=resolved_runtime_path,
    )
    # Force all declarative contracts to validate before runtime construction.
    spec.numerics
    return spec


__all__ = [
    "DEFAULT_FRICTION_MODE_FRIC",
    "DEFAULT_PLANE_RUNTIME_PATH",
    "DEFAULT_SPECTRAL_REFRESH_TIME",
    "DEFAULT_ZERO_MODE_POLICY",
    "PLANE_FREE_SLIP_BOUNDARIES",
    "PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES",
    "PLANE_RUN_SPEC_SCHEMA_VERSION",
    "PlaneBerisEdwardsRunSpec",
    "PlaneFreeSlipBoundaryConditions",
    "PlaneRuntimePath",
    "SpectralRefreshSpec",
    "create_plane_beris_edwards_run_spec",
]
