"""Provisional flat compatibility facade for the legacy Channel request.

P7.1 is declaration-only.  ``Channel.py`` and ``pssolver.channel`` do not
consume this object yet, so constructing a spec cannot start or alter a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
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
from pssolver.geometries import RectangularChannel
from .channel_active_nematics_declarations import (
    CHANNEL_BOUNDARIES,
    ChannelActiveNematicMaterialSpec,
    ChannelBoundaryConditions,
    ChannelExecutionSpec,
    ChannelInitialConditionSpec,
    ChannelPressureSolverSpec,
    ChannelRunComponents,
    ChannelRuntimePath,
    ChannelWorkflowSpec,
    build_channel_stokes_spec,
)


CHANNEL_RUN_SPEC_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChannelActiveNematicRunSpec:
    """Flat P7.1 facade whose defaults describe the current ``Channel.py``."""

    shape: tuple[int, int, int] = (512, 40, 40)
    lengths: tuple[float, float, float] = (128.0, 10.0, 10.0)
    dt: float = 0.01
    steps: int = 1000
    save_interval: int = 10
    diagnostics_enabled: bool = True
    diagnostic_interval: int = 10
    batch_size: int = 1
    seed: int = 24
    rho: float = 6.0
    elastic_constant: float = 1.0
    activity: float = 5.0
    beta: float = -1.0
    friction: float = 0.0
    viscosity: float = 1.0
    flow_alignment: float = 1.0
    initial_s: float = 2.0 / 3.0
    noise_theta: float = 0.01
    noise_phi: float = 0.01
    smoothing_sigma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    initialization_mode: str = "generated"
    snapshot_mode: str = "resume"
    snapshot_directory: Path = Path("data_channel")
    snapshot_step: int = 1000
    generated_output_directory: Path = Path("data_channel")
    snapshot_output_directory: Path = Path("data_channel_snapshot")
    pressure_relative_tolerance: float = 1.0e-6
    pressure_max_iterations: int = 80
    pressure_fixed_iterations: int | None = None
    device: str = "auto"
    dtype: str = "float32"
    dealias_rule: str = "none"
    projected_transform_execution: str = "full"
    transform_execution_order: str = "real_first"
    spectral_storage: str = "full_complex"
    runtime_path: ChannelRuntimePath = ChannelRuntimePath.LEGACY_CHANNEL
    boundaries: ChannelBoundaryConditions = CHANNEL_BOUNDARIES
    _components: ChannelRunComponents = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        try:
            runtime_path = ChannelRuntimePath(self.runtime_path)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported Channel runtime path") from exc
        object.__setattr__(self, "runtime_path", runtime_path)
        components = _build_channel_run_components(self)
        domain = components.geometry.domain
        material = components.material
        initial = components.initial_condition
        execution = components.execution
        workflow = components.workflow
        object.__setattr__(self, "shape", domain.shape)
        object.__setattr__(self, "lengths", domain.lengths)
        object.__setattr__(self, "dt", components.dt)
        object.__setattr__(self, "steps", workflow.steps)
        object.__setattr__(self, "save_interval", workflow.save_interval)
        object.__setattr__(
            self,
            "diagnostic_interval",
            workflow.diagnostic_interval,
        )
        object.__setattr__(self, "batch_size", execution.batch_size)
        object.__setattr__(self, "seed", initial.seed)
        object.__setattr__(self, "rho", material.rho)
        object.__setattr__(self, "elastic_constant", material.elastic_constant)
        object.__setattr__(self, "activity", material.activity)
        object.__setattr__(self, "beta", material.beta)
        object.__setattr__(self, "friction", material.friction)
        object.__setattr__(self, "viscosity", material.viscosity)
        object.__setattr__(self, "flow_alignment", material.flow_alignment)
        object.__setattr__(self, "initial_s", initial.initial_s)
        object.__setattr__(self, "noise_theta", initial.noise_theta)
        object.__setattr__(self, "noise_phi", initial.noise_phi)
        object.__setattr__(self, "smoothing_sigma", initial.smoothing_sigma)
        object.__setattr__(
            self,
            "snapshot_directory",
            initial.snapshot_directory,
        )
        object.__setattr__(
            self,
            "generated_output_directory",
            workflow.generated_output_directory,
        )
        object.__setattr__(
            self,
            "snapshot_output_directory",
            workflow.snapshot_output_directory,
        )
        object.__setattr__(self, "_components", components)

    @property
    def components(self) -> ChannelRunComponents:
        """Return the validated ownership-separated component graph."""

        return self._components

    def to_metadata(self) -> dict[str, object]:
        """Return the resolved declaration without implying CLI connection."""

        return {
            "schema_version": CHANNEL_RUN_SPEC_SCHEMA_VERSION,
            "kind": "channel_active_nematics_run_spec",
            "production_connection": False,
            "package_runtime_facade": True,
            "components": self.components.to_metadata(),
        }

    def canonical_sha256(self) -> str:
        """Hash the complete provisional declaration deterministically."""

        payload = json.dumps(
            self.to_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def runtime_identity_metadata(self) -> dict[str, object]:
        """Return the numerical identity required for exact same-path restart."""

        components = self.components.to_metadata()
        execution = dict(components["execution"])
        execution.pop("device")
        return {
            "schema_version": CHANNEL_RUN_SPEC_SCHEMA_VERSION,
            "authority": (
                "pssolver.configuration.channel_active_nematics."
                "ChannelActiveNematicRunSpec"
            ),
            "runtime_path": self.runtime_path.value,
            "geometry": components["geometry"],
            "boundaries": components["boundaries"],
            "numerics": components["numerics"],
            "material": components["material"],
            "stokes": components["stokes"],
            "pressure_solver": components["pressure_solver"],
            "execution": execution,
            "dt": components["dt"],
        }

    def runtime_identity_sha256(self) -> str:
        payload = json.dumps(
            self.runtime_identity_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def identity_metadata(self) -> dict[str, object]:
        return {
            "schema_version": CHANNEL_RUN_SPEC_SCHEMA_VERSION,
            "authority": (
                "pssolver.configuration.channel_active_nematics."
                "ChannelActiveNematicRunSpec"
            ),
            "runtime_path": self.runtime_path.value,
            "canonical_sha256": self.canonical_sha256(),
            "runtime_identity_sha256": self.runtime_identity_sha256(),
        }

    def runtime_selection_metadata(self) -> dict[str, object]:
        return {
            "authority": (
                "pssolver.configuration.channel_active_nematics."
                "ChannelActiveNematicRunSpec.runtime_path"
            ),
            "default": ChannelRuntimePath.LEGACY_CHANNEL.value,
            "requested": self.runtime_path.value,
            "effective": self.runtime_path.value,
            "fallback_allowed": False,
        }


def _build_channel_run_components(
    spec: ChannelActiveNematicRunSpec,
) -> ChannelRunComponents:
    try:
        domain = DomainSpec(shape=tuple(spec.shape), lengths=tuple(spec.lengths))
    except TypeError as exc:
        raise TypeError("shape and lengths must be iterable") from exc
    if domain.ndim != 3:
        raise ValueError("ChannelActiveNematicRunSpec requires three dimensions")
    geometry = RectangularChannel(domain, streamwise_axis=0)
    try:
        numerics = NumericsConfig(
            precision=Precision(spec.dtype),
            dealias_rule=DealiasRule(spec.dealias_rule),
            projected_transform_execution=ProjectedTransformExecution(
                spec.projected_transform_execution
            ),
            transform_execution_order=TransformExecutionOrder(
                spec.transform_execution_order
            ),
            spectral_storage=SpectralStorage(spec.spectral_storage),
            hermitian_axis=None,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Channel numerical policy") from exc
    if numerics.dealias_rule is not DealiasRule.NONE:
        raise ValueError("P7.1 Channel oracle requires dealias_rule='none'")
    if numerics.projected_transform_execution is not ProjectedTransformExecution.FULL:
        raise ValueError(
            "P7.1 Channel oracle requires projected_transform_execution='full'"
        )
    if numerics.transform_execution_order is not TransformExecutionOrder.REAL_FIRST:
        raise ValueError(
            "P7.1 Channel oracle requires transform_execution_order='real_first'"
        )
    if numerics.spectral_storage is not SpectralStorage.FULL_COMPLEX:
        raise ValueError("P7.1 Channel oracle requires full-complex storage")
    material = ChannelActiveNematicMaterialSpec(
        rho=spec.rho,
        elastic_constant=spec.elastic_constant,
        activity=spec.activity,
        beta=spec.beta,
        friction=spec.friction,
        viscosity=spec.viscosity,
        flow_alignment=spec.flow_alignment,
    )
    stokes = build_channel_stokes_spec(material)
    pressure_solver = ChannelPressureSolverSpec(
        relative_tolerance=spec.pressure_relative_tolerance,
        max_iterations=spec.pressure_max_iterations,
        fixed_iterations=spec.pressure_fixed_iterations,
    )
    initial_condition = ChannelInitialConditionSpec(
        mode=spec.initialization_mode,
        seed=spec.seed,
        initial_s=spec.initial_s,
        noise_theta=spec.noise_theta,
        noise_phi=spec.noise_phi,
        smoothing_sigma=spec.smoothing_sigma,
        snapshot_mode=spec.snapshot_mode,
        snapshot_directory=spec.snapshot_directory,
        snapshot_step=spec.snapshot_step,
    )
    execution = ChannelExecutionSpec(
        device=spec.device,
        dtype=spec.dtype,
        batch_size=spec.batch_size,
        runtime_path=spec.runtime_path,
    )
    workflow = ChannelWorkflowSpec(
        steps=spec.steps,
        save_interval=spec.save_interval,
        diagnostics_enabled=spec.diagnostics_enabled,
        diagnostic_interval=spec.diagnostic_interval,
        generated_output_directory=spec.generated_output_directory,
        snapshot_output_directory=spec.snapshot_output_directory,
    )
    if not isinstance(spec.boundaries, ChannelBoundaryConditions):
        raise TypeError("boundaries must be ChannelBoundaryConditions")
    return ChannelRunComponents(
        geometry=geometry,
        boundaries=spec.boundaries,
        numerics=numerics,
        material=material,
        stokes=stokes,
        pressure_solver=pressure_solver,
        initial_condition=initial_condition,
        execution=execution,
        workflow=workflow,
        dt=spec.dt,
    )


def create_channel_active_nematic_run_spec(
    **overrides: object,
) -> ChannelActiveNematicRunSpec:
    """Construct the provisional flat facade without connecting a runtime."""

    return ChannelActiveNematicRunSpec(**overrides)


__all__ = [
    "CHANNEL_RUN_SPEC_SCHEMA_VERSION",
    "ChannelActiveNematicRunSpec",
    "create_channel_active_nematic_run_spec",
]
