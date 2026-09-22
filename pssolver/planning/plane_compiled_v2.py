"""Tensor-free declarations for the disconnected Plane compiled-v2 path.

P5.1 records the exact production field layout and projected Euler stage
order without importing Torch, allocating tensors, or connecting a runtime
selector.  Tensor binding belongs to P5.2 and execution belongs to P5.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


PLANE_COMPILED_V2_IDENTITY = "compiled_v2"
PLANE_COMPILED_V2_SCHEMA_VERSION = 1


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{label} must be a Python identifier")
    return value


def _require_unique_identifiers(
    values: object,
    label: str,
    *,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    if nonempty and not values:
        raise ValueError(f"{label} must be nonempty")
    normalized = tuple(
        _require_identifier(value, f"{label} entry") for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must contain unique identifiers")
    return normalized


class PlaneFieldRole(str, Enum):
    """Ownership role of one Plane field in the compiled declaration."""

    EVOLVED = "evolved"
    ALGEBRAIC = "algebraic"


class PlaneTransformFamily(str, Enum):
    """Three-axis basis family used by one packed component group."""

    FFT_FFT_DCT = "fft_fft_dct"
    FFT_FFT_DST = "fft_fft_dst"


class PlaneCompiledStage(str, Enum):
    """Frozen projected semi-implicit Euler operation order."""

    PREPARE_ALGEBRAIC = "prepare_algebraic"
    PRE_UPDATE_CALLBACK = "pre_update_callback"
    EXPLICIT_RHS = "explicit_rhs"
    SPECTRAL_ADD_DT_RHS = "spectral_add_dt_rhs"
    SPECTRAL_DIVIDE_BY_DENOMINATOR = (
        "spectral_divide_by_denominator"
    )
    PROJECT_DYNAMIC_SPECTRA = "project_dynamic_spectra"
    INVERSE_DYNAMIC_SPECTRA = "inverse_dynamic_spectra"
    SCHEDULED_SPECTRAL_REFRESH = "scheduled_spectral_refresh"
    COMMIT_PROGRESS = "commit_progress"


class PlaneWorkspaceLifetime(str, Enum):
    """Lifetime of a predeclared workspace category."""

    RUNTIME = "runtime"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class PlaneTransformGroupDeclaration:
    """One packed field group sharing role and transform family."""

    name: str
    components: tuple[str, ...]
    role: PlaneFieldRole
    transform: PlaneTransformFamily

    def __post_init__(self) -> None:
        _require_identifier(self.name, "transform group name")
        _require_unique_identifiers(
            self.components,
            "transform group components",
        )
        if not isinstance(self.role, PlaneFieldRole):
            raise TypeError("transform group role must be a PlaneFieldRole")
        if not isinstance(self.transform, PlaneTransformFamily):
            raise TypeError(
                "transform group transform must be a PlaneTransformFamily"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "components": list(self.components),
            "role": self.role.value,
            "transform": self.transform.value,
        }


@dataclass(frozen=True, slots=True)
class PlaneFieldLayoutDeclaration:
    """Complete component ordering and transform partition for Plane."""

    evolved_components: tuple[str, ...]
    algebraic_components: tuple[str, ...]
    transform_groups: tuple[PlaneTransformGroupDeclaration, ...]

    def __post_init__(self) -> None:
        evolved = _require_unique_identifiers(
            self.evolved_components,
            "evolved components",
        )
        algebraic = _require_unique_identifiers(
            self.algebraic_components,
            "algebraic components",
        )
        if set(evolved).intersection(algebraic):
            raise ValueError(
                "evolved and algebraic components must be disjoint"
            )
        if not isinstance(self.transform_groups, tuple):
            raise TypeError("transform_groups must be a tuple")
        if not self.transform_groups:
            raise ValueError("transform_groups must be nonempty")
        if not all(
            isinstance(group, PlaneTransformGroupDeclaration)
            for group in self.transform_groups
        ):
            raise TypeError(
                "transform_groups must contain transform declarations"
            )
        names = tuple(group.name for group in self.transform_groups)
        if len(set(names)) != len(names):
            raise ValueError("transform group names must be unique")
        grouped = tuple(
            component
            for group in self.transform_groups
            for component in group.components
        )
        expected = evolved + algebraic
        if len(set(grouped)) != len(grouped):
            raise ValueError(
                "each component must occur in exactly one transform group"
            )
        if set(grouped) != set(expected):
            raise ValueError(
                "transform groups must exactly cover declared components"
            )
        role_by_component = {
            component: group.role
            for group in self.transform_groups
            for component in group.components
        }
        if any(
            role_by_component[component] is not PlaneFieldRole.EVOLVED
            for component in evolved
        ):
            raise ValueError("evolved components must use the evolved role")
        if any(
            role_by_component[component] is not PlaneFieldRole.ALGEBRAIC
            for component in algebraic
        ):
            raise ValueError(
                "algebraic components must use the algebraic role"
            )

    @property
    def component_order(self) -> tuple[str, ...]:
        return self.evolved_components + self.algebraic_components

    def to_metadata(self) -> dict[str, object]:
        return {
            "evolved_components": list(self.evolved_components),
            "algebraic_components": list(self.algebraic_components),
            "component_order": list(self.component_order),
            "transform_groups": [
                group.to_metadata() for group in self.transform_groups
            ],
        }


@dataclass(frozen=True, slots=True)
class PlanePersistentStateDeclaration:
    """Persistent state required across projected Euler timesteps."""

    integrator: str
    history_depth: int
    items: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.integrator, "integrator")
        if (
            not isinstance(self.history_depth, int)
            or isinstance(self.history_depth, bool)
            or self.history_depth < 0
        ):
            raise ValueError("history_depth must be a non-negative integer")
        _require_unique_identifiers(self.items, "persistent state items")

    def to_metadata(self) -> dict[str, object]:
        return {
            "integrator": self.integrator,
            "history_depth": self.history_depth,
            "items": list(self.items),
        }


@dataclass(frozen=True, slots=True)
class PlaneWorkspaceRequirement:
    """Semantic workspace category; P5.2 binds shape and storage."""

    name: str
    owner: str
    lifetime: PlaneWorkspaceLifetime
    bounded: bool = True
    construction_bound: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.name, "workspace name")
        _require_identifier(self.owner, "workspace owner")
        if not isinstance(self.lifetime, PlaneWorkspaceLifetime):
            raise TypeError(
                "workspace lifetime must be a PlaneWorkspaceLifetime"
            )
        if not isinstance(self.bounded, bool):
            raise TypeError("workspace bounded must be a bool")
        if not isinstance(self.construction_bound, bool):
            raise TypeError("workspace construction_bound must be a bool")
        if not self.bounded or not self.construction_bound:
            raise ValueError(
                "compiled-v2 workspaces must be bounded and construction-bound"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "owner": self.owner,
            "lifetime": self.lifetime.value,
            "bounded": self.bounded,
            "construction_bound": self.construction_bound,
        }


@dataclass(frozen=True, slots=True)
class PlaneCompiledDeclaration:
    """Disconnected tensor-free declaration for Plane compiled-v2."""

    identity: str
    schema_version: int
    layout: PlaneFieldLayoutDeclaration
    persistent_state: PlanePersistentStateDeclaration
    operation_order: tuple[PlaneCompiledStage, ...]
    workspace_requirements: tuple[PlaneWorkspaceRequirement, ...]
    hot_loop_forbidden: tuple[str, ...]
    connected_runtime: None = None

    def __post_init__(self) -> None:
        if self.identity != PLANE_COMPILED_V2_IDENTITY:
            raise ValueError("compiled declaration identity must be compiled_v2")
        if self.schema_version != PLANE_COMPILED_V2_SCHEMA_VERSION:
            raise ValueError("unsupported compiled declaration schema version")
        if not isinstance(self.layout, PlaneFieldLayoutDeclaration):
            raise TypeError("layout must be a PlaneFieldLayoutDeclaration")
        if not isinstance(
            self.persistent_state,
            PlanePersistentStateDeclaration,
        ):
            raise TypeError(
                "persistent_state must be a persistent state declaration"
            )
        if not isinstance(self.operation_order, tuple) or not all(
            isinstance(stage, PlaneCompiledStage)
            for stage in self.operation_order
        ):
            raise TypeError(
                "operation_order must contain PlaneCompiledStage values"
            )
        if self.operation_order != _PROJECTED_EULER_STAGE_ORDER:
            raise ValueError(
                "operation_order must match projected Euler production order"
            )
        if not isinstance(self.workspace_requirements, tuple) or not all(
            isinstance(requirement, PlaneWorkspaceRequirement)
            for requirement in self.workspace_requirements
        ):
            raise TypeError(
                "workspace_requirements must contain workspace requirements"
            )
        workspace_names = tuple(
            requirement.name for requirement in self.workspace_requirements
        )
        if len(set(workspace_names)) != len(workspace_names):
            raise ValueError("workspace requirement names must be unique")
        _require_unique_identifiers(
            self.hot_loop_forbidden,
            "hot-loop forbidden operations",
        )
        if self.connected_runtime is not None:
            raise ValueError("P5.1 declarations must remain disconnected")

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "layout": self.layout.to_metadata(),
            "persistent_state": self.persistent_state.to_metadata(),
            "operation_order": [
                stage.value for stage in self.operation_order
            ],
            "workspace_requirements": [
                requirement.to_metadata()
                for requirement in self.workspace_requirements
            ],
            "hot_loop_forbidden": list(self.hot_loop_forbidden),
            "connected_runtime": self.connected_runtime,
        }


_PROJECTED_EULER_STAGE_ORDER = (
    PlaneCompiledStage.PREPARE_ALGEBRAIC,
    PlaneCompiledStage.PRE_UPDATE_CALLBACK,
    PlaneCompiledStage.EXPLICIT_RHS,
    PlaneCompiledStage.SPECTRAL_ADD_DT_RHS,
    PlaneCompiledStage.SPECTRAL_DIVIDE_BY_DENOMINATOR,
    PlaneCompiledStage.PROJECT_DYNAMIC_SPECTRA,
    PlaneCompiledStage.INVERSE_DYNAMIC_SPECTRA,
    PlaneCompiledStage.SCHEDULED_SPECTRAL_REFRESH,
    PlaneCompiledStage.COMMIT_PROGRESS,
)


def plane_compiled_v2_declaration() -> PlaneCompiledDeclaration:
    """Return the canonical disconnected P5.1 declaration."""

    q_components = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    algebraic_components = ("ux", "uy", "uz", "p")
    return PlaneCompiledDeclaration(
        identity=PLANE_COMPILED_V2_IDENTITY,
        schema_version=PLANE_COMPILED_V2_SCHEMA_VERSION,
        layout=PlaneFieldLayoutDeclaration(
            evolved_components=q_components,
            algebraic_components=algebraic_components,
            transform_groups=(
                PlaneTransformGroupDeclaration(
                    name="q_neumann",
                    components=q_components,
                    role=PlaneFieldRole.EVOLVED,
                    transform=PlaneTransformFamily.FFT_FFT_DCT,
                ),
                PlaneTransformGroupDeclaration(
                    name="tangential_velocity_neumann",
                    components=("ux", "uy"),
                    role=PlaneFieldRole.ALGEBRAIC,
                    transform=PlaneTransformFamily.FFT_FFT_DCT,
                ),
                PlaneTransformGroupDeclaration(
                    name="normal_velocity_dirichlet",
                    components=("uz",),
                    role=PlaneFieldRole.ALGEBRAIC,
                    transform=PlaneTransformFamily.FFT_FFT_DST,
                ),
                PlaneTransformGroupDeclaration(
                    name="pressure_modal_neumann",
                    components=("p",),
                    role=PlaneFieldRole.ALGEBRAIC,
                    transform=PlaneTransformFamily.FFT_FFT_DCT,
                ),
            ),
        ),
        persistent_state=PlanePersistentStateDeclaration(
            integrator="projected_semi_implicit_euler",
            history_depth=0,
            items=(
                "evolved_physical",
                "evolved_spectral",
                "completed_steps",
                "spectral_refresh_interval",
                "spectral_refresh_step_count",
                "spectral_refresh_count",
                "representation_ledger",
            ),
        ),
        operation_order=_PROJECTED_EULER_STAGE_ORDER,
        workspace_requirements=(
            PlaneWorkspaceRequirement(
                name="explicit_q_rhs",
                owner="integrator",
                lifetime=PlaneWorkspaceLifetime.GENERATION,
            ),
            PlaneWorkspaceRequirement(
                name="q_gradient_cache",
                owner="constitutive",
                lifetime=PlaneWorkspaceLifetime.RUNTIME,
            ),
            PlaneWorkspaceRequirement(
                name="constitutive_scratch",
                owner="constitutive",
                lifetime=PlaneWorkspaceLifetime.GENERATION,
            ),
            PlaneWorkspaceRequirement(
                name="nematic_force",
                owner="stokes",
                lifetime=PlaneWorkspaceLifetime.GENERATION,
            ),
            PlaneWorkspaceRequirement(
                name="stokes_solve_scratch",
                owner="stokes",
                lifetime=PlaneWorkspaceLifetime.GENERATION,
            ),
            PlaneWorkspaceRequirement(
                name="projected_transform_scratch",
                owner="projector",
                lifetime=PlaneWorkspaceLifetime.RUNTIME,
            ),
            PlaneWorkspaceRequirement(
                name="inverse_transform_scratch",
                owner="projector",
                lifetime=PlaneWorkspaceLifetime.RUNTIME,
            ),
        ),
        hot_loop_forbidden=(
            "registry_lookup",
            "capability_lookup",
            "string_field_discovery",
            "metadata_construction",
            "configuration_parsing",
            "global_device_inference",
            "global_dtype_inference",
            "unbounded_tensor_allocation",
            "runtime_fallback",
        ),
    )


__all__ = [
    "PLANE_COMPILED_V2_IDENTITY",
    "PLANE_COMPILED_V2_SCHEMA_VERSION",
    "PlaneCompiledDeclaration",
    "PlaneCompiledStage",
    "PlaneFieldLayoutDeclaration",
    "PlaneFieldRole",
    "PlanePersistentStateDeclaration",
    "PlaneTransformFamily",
    "PlaneTransformGroupDeclaration",
    "PlaneWorkspaceLifetime",
    "PlaneWorkspaceRequirement",
    "plane_compiled_v2_declaration",
]
