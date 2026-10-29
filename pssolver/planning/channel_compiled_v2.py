"""Tensor-free P7.3 declarations for opt-in compiled Channel execution.

The declaration fixes ownership, modal parity, persistent pressure state and
the timestep operation order.  It imports no tensor runtime and does not add a
production runtime selector; construction is available only through the
direct experimental adapter named in :func:`channel_compiled_v2_declaration`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


CHANNEL_COMPILED_V2_IDENTITY = "compiled_channel_v2"
CHANNEL_COMPILED_V2_SCHEMA_VERSION = 1


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


def _identifiers(values: object, description: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{description} must be a nonempty tuple")
    normalized = tuple(
        _identifier(value, f"{description} entry") for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{description} must contain unique identifiers")
    return normalized


class ChannelFieldRole(str, Enum):
    """Persistent field role in the compiled Channel layout."""

    EVOLVED = "evolved"
    ALGEBRAIC = "algebraic"


class ChannelTransformFamily(str, Enum):
    """Native transform family for one Channel component group."""

    FFT_DCT_DCT = "fft_dct_dct"
    FFT_DST_DST = "fft_dst_dst"


class ChannelCompiledStage(str, Enum):
    """Frozen semi-implicit Euler operation order."""

    PREPARE_ALGEBRAIC = "prepare_algebraic"
    PRE_UPDATE_CALLBACK = "pre_update_callback"
    EXPLICIT_RHS = "explicit_rhs"
    SPECTRAL_ADD_DT_RHS = "spectral_add_dt_rhs"
    SPECTRAL_DIVIDE_BY_DENOMINATOR = "spectral_divide_by_denominator"
    PROJECT_DYNAMIC_SPECTRA = "project_dynamic_spectra"
    INVERSE_DYNAMIC_SPECTRA = "inverse_dynamic_spectra"
    SCHEDULED_SPECTRAL_REFRESH = "scheduled_spectral_refresh"
    COMMIT_PROGRESS = "commit_progress"


class ChannelWorkspaceLifetime(str, Enum):
    """Lifetime of one semantically bounded workspace category."""

    RUNTIME = "runtime"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class ChannelTransformGroupDeclaration:
    """One component group sharing role and modal parity."""

    name: str
    components: tuple[str, ...]
    role: ChannelFieldRole
    transform: ChannelTransformFamily

    def __post_init__(self) -> None:
        _identifier(self.name, "transform-group name")
        _identifiers(self.components, "transform-group components")
        if not isinstance(self.role, ChannelFieldRole):
            raise TypeError("role must be a ChannelFieldRole")
        if not isinstance(self.transform, ChannelTransformFamily):
            raise TypeError("transform must be a ChannelTransformFamily")

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "components": list(self.components),
            "role": self.role.value,
            "transform": self.transform.value,
        }


@dataclass(frozen=True, slots=True)
class ChannelPersistentStateDeclaration:
    """State that survives a successful Channel timestep."""

    integrator: str
    history_depth: int
    items: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.integrator, "integrator")
        if (
            not isinstance(self.history_depth, int)
            or isinstance(self.history_depth, bool)
            or self.history_depth < 0
        ):
            raise ValueError("history_depth must be a non-negative integer")
        _identifiers(self.items, "persistent-state items")

    def to_metadata(self) -> dict[str, object]:
        return {
            "integrator": self.integrator,
            "history_depth": self.history_depth,
            "items": list(self.items),
        }


@dataclass(frozen=True, slots=True)
class ChannelWorkspaceRequirement:
    """Construction-bound semantic workspace category."""

    name: str
    owner: str
    lifetime: ChannelWorkspaceLifetime
    bounded: bool = True
    construction_bound: bool = True

    def __post_init__(self) -> None:
        _identifier(self.name, "workspace name")
        _identifier(self.owner, "workspace owner")
        if not isinstance(self.lifetime, ChannelWorkspaceLifetime):
            raise TypeError("lifetime must be a ChannelWorkspaceLifetime")
        if not isinstance(self.bounded, bool):
            raise TypeError("bounded must be a bool")
        if not isinstance(self.construction_bound, bool):
            raise TypeError("construction_bound must be a bool")
        if not self.bounded or not self.construction_bound:
            raise ValueError(
                "compiled Channel workspaces must be bounded and "
                "construction-bound"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "owner": self.owner,
            "lifetime": self.lifetime.value,
            "bounded": self.bounded,
            "construction_bound": self.construction_bound,
        }


_STAGE_ORDER = (
    ChannelCompiledStage.PREPARE_ALGEBRAIC,
    ChannelCompiledStage.PRE_UPDATE_CALLBACK,
    ChannelCompiledStage.EXPLICIT_RHS,
    ChannelCompiledStage.SPECTRAL_ADD_DT_RHS,
    ChannelCompiledStage.SPECTRAL_DIVIDE_BY_DENOMINATOR,
    ChannelCompiledStage.PROJECT_DYNAMIC_SPECTRA,
    ChannelCompiledStage.INVERSE_DYNAMIC_SPECTRA,
    ChannelCompiledStage.SCHEDULED_SPECTRAL_REFRESH,
    ChannelCompiledStage.COMMIT_PROGRESS,
)


@dataclass(frozen=True, slots=True)
class ChannelCompiledV2Declaration:
    """Complete P7.3 Channel declaration and direct execution identity."""

    identity: str
    schema_version: int
    evolved_components: tuple[str, ...]
    algebraic_components: tuple[str, ...]
    transform_groups: tuple[ChannelTransformGroupDeclaration, ...]
    persistent_state: ChannelPersistentStateDeclaration
    operation_order: tuple[ChannelCompiledStage, ...]
    workspace_requirements: tuple[ChannelWorkspaceRequirement, ...]
    hot_loop_forbidden: tuple[str, ...]
    execution_adapter: str
    production_connection: bool = False

    def __post_init__(self) -> None:
        if self.identity != CHANNEL_COMPILED_V2_IDENTITY:
            raise ValueError("invalid compiled Channel identity")
        if self.schema_version != CHANNEL_COMPILED_V2_SCHEMA_VERSION:
            raise ValueError("unsupported compiled Channel schema version")
        evolved = _identifiers(
            self.evolved_components,
            "evolved components",
        )
        algebraic = _identifiers(
            self.algebraic_components,
            "algebraic components",
        )
        if set(evolved).intersection(algebraic):
            raise ValueError("evolved and algebraic components must be disjoint")
        if not isinstance(self.transform_groups, tuple) or not all(
            isinstance(group, ChannelTransformGroupDeclaration)
            for group in self.transform_groups
        ):
            raise TypeError("transform_groups must contain declarations")
        grouped = tuple(
            component
            for group in self.transform_groups
            for component in group.components
        )
        if len(set(grouped)) != len(grouped):
            raise ValueError("each component must occur in one transform group")
        if set(grouped) != set(evolved + algebraic):
            raise ValueError("transform groups must exactly cover all components")
        roles = {
            component: group.role
            for group in self.transform_groups
            for component in group.components
        }
        if any(roles[name] is not ChannelFieldRole.EVOLVED for name in evolved):
            raise ValueError("evolved components must use the evolved role")
        if any(
            roles[name] is not ChannelFieldRole.ALGEBRAIC for name in algebraic
        ):
            raise ValueError("algebraic components must use the algebraic role")
        if not isinstance(
            self.persistent_state,
            ChannelPersistentStateDeclaration,
        ):
            raise TypeError("persistent_state has the wrong type")
        if self.operation_order != _STAGE_ORDER:
            raise ValueError("operation_order must match the Channel oracle")
        if not isinstance(self.workspace_requirements, tuple) or not all(
            isinstance(item, ChannelWorkspaceRequirement)
            for item in self.workspace_requirements
        ):
            raise TypeError("workspace_requirements have the wrong type")
        workspace_names = tuple(
            item.name for item in self.workspace_requirements
        )
        if len(set(workspace_names)) != len(workspace_names):
            raise ValueError("workspace requirement names must be unique")
        _identifiers(self.hot_loop_forbidden, "hot-loop forbidden operations")
        if self.execution_adapter != (
            "pssolver.experimental.channel_compiled_v2"
        ):
            raise ValueError("execution_adapter must name the P7.3 adapter")
        if self.production_connection is not False:
            raise ValueError("P7.3 must remain disconnected from production")

    @property
    def component_order(self) -> tuple[str, ...]:
        return self.evolved_components + self.algebraic_components

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "component_order": list(self.component_order),
            "evolved_components": list(self.evolved_components),
            "algebraic_components": list(self.algebraic_components),
            "transform_groups": [
                group.to_metadata() for group in self.transform_groups
            ],
            "persistent_state": self.persistent_state.to_metadata(),
            "operation_order": [
                stage.value for stage in self.operation_order
            ],
            "workspace_requirements": [
                item.to_metadata() for item in self.workspace_requirements
            ],
            "hot_loop_forbidden": list(self.hot_loop_forbidden),
            "execution_adapter": self.execution_adapter,
            "production_connection": self.production_connection,
        }


def channel_compiled_v2_declaration() -> ChannelCompiledV2Declaration:
    """Return the canonical tensor-free P7.3 Channel declaration."""

    q_components = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    algebraic_components = ("ux", "uy", "uz", "p")
    return ChannelCompiledV2Declaration(
        identity=CHANNEL_COMPILED_V2_IDENTITY,
        schema_version=CHANNEL_COMPILED_V2_SCHEMA_VERSION,
        evolved_components=q_components,
        algebraic_components=algebraic_components,
        transform_groups=(
            ChannelTransformGroupDeclaration(
                name="q_neumann_neumann",
                components=q_components,
                role=ChannelFieldRole.EVOLVED,
                transform=ChannelTransformFamily.FFT_DCT_DCT,
            ),
            ChannelTransformGroupDeclaration(
                name="velocity_dirichlet_dirichlet",
                components=("ux", "uy", "uz"),
                role=ChannelFieldRole.ALGEBRAIC,
                transform=ChannelTransformFamily.FFT_DST_DST,
            ),
            ChannelTransformGroupDeclaration(
                name="pressure_neumann_neumann",
                components=("p",),
                role=ChannelFieldRole.ALGEBRAIC,
                transform=ChannelTransformFamily.FFT_DCT_DCT,
            ),
        ),
        persistent_state=ChannelPersistentStateDeclaration(
            integrator="semi_implicit_euler",
            history_depth=0,
            items=(
                "evolved_physical",
                "evolved_spectral",
                "completed_steps",
                "spectral_refresh_interval",
                "spectral_refresh_step_count",
                "spectral_refresh_count",
                "representation_ledger",
                "pressure_guess",
                "pressure_iterations",
                "pressure_residual",
                "pressure_relative_residual",
            ),
        ),
        operation_order=_STAGE_ORDER,
        workspace_requirements=tuple(
            ChannelWorkspaceRequirement(name, owner, lifetime)
            for name, owner, lifetime in (
                (
                    "explicit_q_rhs",
                    "integrator",
                    ChannelWorkspaceLifetime.GENERATION,
                ),
                (
                    "constitutive_scratch",
                    "constitutive",
                    ChannelWorkspaceLifetime.GENERATION,
                ),
                (
                    "active_force",
                    "stokes",
                    ChannelWorkspaceLifetime.GENERATION,
                ),
                (
                    "bounded_axis_basis_change",
                    "stokes",
                    ChannelWorkspaceLifetime.RUNTIME,
                ),
                (
                    "pressure_pcg_vectors",
                    "stokes",
                    ChannelWorkspaceLifetime.RUNTIME,
                ),
                (
                    "inverse_transform_scratch",
                    "transforms",
                    ChannelWorkspaceLifetime.RUNTIME,
                ),
            )
        ),
        hot_loop_forbidden=(
            "registry_lookup",
            "capability_lookup",
            "string_field_discovery",
            "metadata_construction",
            "configuration_parsing",
            "global_device_inference",
            "global_dtype_inference",
            "runtime_fallback",
        ),
        execution_adapter="pssolver.experimental.channel_compiled_v2",
    )


__all__ = [
    "CHANNEL_COMPILED_V2_IDENTITY",
    "CHANNEL_COMPILED_V2_SCHEMA_VERSION",
    "ChannelCompiledStage",
    "ChannelCompiledV2Declaration",
    "ChannelFieldRole",
    "ChannelPersistentStateDeclaration",
    "ChannelTransformFamily",
    "ChannelTransformGroupDeclaration",
    "ChannelWorkspaceLifetime",
    "ChannelWorkspaceRequirement",
    "channel_compiled_v2_declaration",
]
