"""Private construction-time bindings for the disconnected Plane runtime.

P5.2 audits and adopts the exact tensor storage and operators created by the
qualified legacy Plane assembly.  It neither installs a runtime selector nor
executes a timestep.  P5.3 may consume this immutable binding plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import string

import torch

from pssolver.configuration import PlaneBerisEdwardsRunSpec, PlaneRuntimePath
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.execution.state import CurrentRepresentation, RuntimeState
from pssolver.execution.workspace import RuntimeWorkspace
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsQNonlinearModel,
)
from pssolver.operators.projection import BasisAwareSpectralProjector
from pssolver.planning.plane_compiled_v2 import (
    PlaneCompiledDeclaration,
    PlaneCompiledStage,
    PlaneFieldRole,
    PlaneTransformFamily,
    plane_compiled_v2_declaration,
)
from pssolver.runtime.plane_legacy import (
    DealiasedSemiImplicitEulerIntegrator,
)


def _tensor_storage_identity(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "storage_offset": int(tensor.storage_offset()),
        "storage_bytes": int(storage.nbytes()),
    }


def _require_tensor(
    tensor: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    finite: bool = True,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}, expected {shape}"
        )
    if tensor.dtype != dtype:
        raise ValueError(
            f"{name} has dtype {tensor.dtype}, expected {dtype}"
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} is on {tensor.device}, expected {device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if finite and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _require_same_storage(
    value: torch.Tensor,
    owner: torch.Tensor,
    *,
    name: str,
) -> None:
    if _tensor_storage_identity(value) != _tensor_storage_identity(owner):
        raise ValueError(f"{name} must be a zero-copy view of owned storage")


def _require_distinct_storage(
    tensors: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    identities: dict[tuple[str, int, int], str] = {}
    for name, tensor in tensors:
        identity = _tensor_storage_identity(tensor)
        if identity in identities:
            raise ValueError(
                f"unexpected tensor alias between {identities[identity]} "
                f"and {name}"
            )
        identities[identity] = name


def _require_close(actual: object, expected: object, name: str) -> float:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(actual_value) or actual_value != expected_value:
        raise ValueError(
            f"{name} is {actual_value!r}, expected {expected_value!r}"
        )
    return actual_value


@dataclass(frozen=True, slots=True)
class PlaneBoundTransformGroup:
    """One semantic or execution transform group bound to field indices."""

    name: str
    components: tuple[str, ...]
    indices: tuple[int, ...]
    boundary_conditions: tuple[str, ...]
    transform_kinds: tuple[str, ...]
    indexing_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("bound transform group name must be an identifier")
        if not self.components or len(set(self.components)) != len(
            self.components
        ):
            raise ValueError("bound transform components must be nonempty and unique")
        if len(self.components) != len(self.indices):
            raise ValueError("bound transform components and indices must align")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            for index in self.indices
        ):
            raise ValueError("bound transform indices must be non-negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("bound transform indices must be unique")
        if len(self.boundary_conditions) != 3:
            raise ValueError("Plane boundary conditions must have three axes")
        if len(self.transform_kinds) != 3:
            raise ValueError("Plane transform kinds must have three axes")
        if self.indexing_mode not in {"contiguous_slice", "advanced"}:
            raise ValueError("unsupported transform-group indexing mode")

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "components": list(self.components),
            "indices": list(self.indices),
            "boundary_conditions": list(self.boundary_conditions),
            "transform_kinds": list(self.transform_kinds),
            "indexing_mode": self.indexing_mode,
        }


@dataclass(frozen=True, slots=True)
class PlaneScalarBinding:
    """One immutable scalar and its construction authority."""

    name: str
    value: float
    authority: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("scalar binding name must be an identifier")
        if not math.isfinite(self.value):
            raise ValueError("scalar binding value must be finite")
        if not isinstance(self.authority, str) or not self.authority:
            raise ValueError("scalar binding authority must be nonempty")

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "authority": self.authority,
        }


@dataclass(frozen=True, slots=True)
class PlaneCompiledTensorBindings:
    """Zero-copy packed tensor views adopted from the production solver."""

    evolved_physical: torch.Tensor
    algebraic_physical: torch.Tensor
    evolved_spectral: torch.Tensor
    algebraic_spectral: torch.Tensor
    linear_operator: torch.Tensor
    denominator: torch.Tensor
    activity: torch.Tensor

    def to_metadata(self) -> dict[str, object]:
        return {
            "evolved_physical": _tensor_metadata(self.evolved_physical),
            "algebraic_physical": _tensor_metadata(self.algebraic_physical),
            "evolved_spectral": _tensor_metadata(self.evolved_spectral),
            "algebraic_spectral": _tensor_metadata(self.algebraic_spectral),
            "linear_operator": _tensor_metadata(self.linear_operator),
            "denominator": _tensor_metadata(self.denominator),
            "activity": _tensor_metadata(self.activity),
            "all_field_views_are_zero_copy": True,
        }


@dataclass(frozen=True, slots=True)
class PlaneCompiledOperatorBindings:
    """Pre-resolved production operators; no lookup is needed per step."""

    prepare_algebraic: Callable[..., object]
    explicit_rhs: Callable[..., object]
    project_dynamic_spectra: Callable[..., object]
    inverse_dynamic_spectra: Callable[..., object]
    refresh_dynamic_spectra: Callable[..., object]
    q_rhs_kernel: BerisEdwardsQNonlinearModel
    stokes_kernel: BerisEdwardsFreeSlipStokes
    projector: BasisAwareSpectralProjector

    def __post_init__(self) -> None:
        for name in (
            "prepare_algebraic",
            "explicit_rhs",
            "project_dynamic_spectra",
            "inverse_dynamic_spectra",
            "refresh_dynamic_spectra",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def to_metadata(self) -> dict[str, object]:
        return {
            "prepare_algebraic": (
                "DealiasedSemiImplicitEulerIntegrator._prepare_algebraic"
            ),
            "explicit_rhs": (
                "DealiasedSemiImplicitEulerIntegrator._explicit_rhs"
            ),
            "project_dynamic_spectra": (
                "DealiasedSemiImplicitEulerIntegrator._project_dynamic_spectra"
            ),
            "inverse_dynamic_spectra": (
                "DealiasedSemiImplicitEulerIntegrator._inverse_dynamic_spectra"
            ),
            "refresh_dynamic_spectra": (
                "DealiasedSemiImplicitEulerIntegrator."
                "_refresh_dynamic_spectra_operation"
            ),
            "q_rhs_kernel": type(self.q_rhs_kernel).__name__,
            "stokes_kernel": type(self.stokes_kernel).__name__,
            "projector": type(self.projector).__name__,
            "all_operators_prebound": True,
        }


@dataclass(frozen=True, slots=True)
class PlaneCompiledV2BindingPlan:
    """Immutable disconnected result of the P5.2 construction audit."""

    declaration: PlaneCompiledDeclaration
    runtime_identity_sha256: str
    fields: object
    state: RuntimeState
    tensors: PlaneCompiledTensorBindings
    operators: PlaneCompiledOperatorBindings
    scalar_bindings: tuple[PlaneScalarBinding, ...]
    semantic_transform_groups: tuple[PlaneBoundTransformGroup, ...]
    execution_transform_groups: tuple[PlaneBoundTransformGroup, ...]
    legacy_workspace_bytes: int
    additional_workspace_bytes: int
    connected_runtime: None = None

    def __post_init__(self) -> None:
        if self.declaration.connected_runtime is not None:
            raise ValueError("compiled declaration must remain disconnected")
        if (
            not isinstance(self.runtime_identity_sha256, str)
            or len(self.runtime_identity_sha256) != 64
            or any(
                character not in string.hexdigits
                for character in self.runtime_identity_sha256
            )
        ):
            raise ValueError("runtime identity must be a SHA-256 hex string")
        scalar_names = tuple(binding.name for binding in self.scalar_bindings)
        if len(set(scalar_names)) != len(scalar_names):
            raise ValueError("scalar bindings must be unique")
        if self.legacy_workspace_bytes != 0:
            raise ValueError("legacy assembly retained a nonempty workspace")
        if self.additional_workspace_bytes != 0:
            raise ValueError("P5.2 may not allocate an additional workspace")
        if self.connected_runtime is not None:
            raise ValueError("P5.2 binding plan must remain disconnected")

    def to_metadata(self) -> dict[str, object]:
        root_storages = {
            _tensor_storage_identity(tensor)
            for tensor in (
                self.fields.spatial,
                self.fields.spectral,
                self.fields.L_hat,
                self.tensors.denominator,
                self.tensors.activity,
            )
        }
        return {
            "schema_version": 1,
            "identity": self.declaration.identity,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "declaration": self.declaration.to_metadata(),
            "tensors": self.tensors.to_metadata(),
            "operators": self.operators.to_metadata(),
            "scalar_bindings": [
                binding.to_metadata() for binding in self.scalar_bindings
            ],
            "semantic_transform_groups": [
                group.to_metadata() for group in self.semantic_transform_groups
            ],
            "execution_transform_groups": [
                group.to_metadata() for group in self.execution_transform_groups
            ],
            "storage": {
                "owned_root_storage_count": len(root_storages),
                "owned_root_storage_bytes": sum(
                    identity[2] for identity in root_storages
                ),
                "reuses_existing_solver_storage": True,
                "legacy_workspace_bytes": self.legacy_workspace_bytes,
                "additional_workspace_bytes": self.additional_workspace_bytes,
                "duplicate_workspace_retained": False,
            },
            "connected_runtime": self.connected_runtime,
        }


def _bind_transform_group(
    fields: object,
    *,
    name: str,
    components: tuple[str, ...],
) -> PlaneBoundTransformGroup:
    indices = tuple(fields.name_to_idx[component] for component in components)
    boundary_conditions = tuple(fields.get_boundary_conditions(indices[0]))
    transform_kinds = tuple(fields.get_transform_plan(indices[0]))
    for index in indices[1:]:
        if tuple(fields.get_boundary_conditions(index)) != boundary_conditions:
            raise ValueError(f"transform group {name} has mixed boundaries")
        if tuple(fields.get_transform_plan(index)) != transform_kinds:
            raise ValueError(f"transform group {name} has mixed transforms")
    indexing = fields.transform_group_indexing_metadata(indices)["effective"]
    return PlaneBoundTransformGroup(
        name=name,
        components=components,
        indices=indices,
        boundary_conditions=boundary_conditions,
        transform_kinds=transform_kinds,
        indexing_mode=indexing,
    )


def _validate_semantic_groups(
    declaration: PlaneCompiledDeclaration,
    groups: tuple[PlaneBoundTransformGroup, ...],
) -> None:
    if tuple(group.name for group in groups) != tuple(
        group.name for group in declaration.layout.transform_groups
    ):
        raise ValueError("semantic transform group identity mismatch")
    expected_kinds = {
        PlaneTransformFamily.FFT_FFT_DCT: ("fft", "fft", "dct"),
        PlaneTransformFamily.FFT_FFT_DST: ("fft", "fft", "dst"),
    }
    for declared, bound in zip(
        declaration.layout.transform_groups,
        groups,
        strict=True,
    ):
        if bound.components != declared.components:
            raise ValueError("semantic transform component mismatch")
        if bound.transform_kinds != expected_kinds[declared.transform]:
            raise ValueError("semantic transform basis mismatch")


def _validate_operator_graph(
    solver: object,
    projector: BasisAwareSpectralProjector,
) -> PlaneCompiledOperatorBindings:
    model = solver.model
    integrator = solver.integrator
    nonlinear = model.nlmodel
    stokes = model.static_model
    if type(integrator) is not DealiasedSemiImplicitEulerIntegrator:
        raise TypeError("P5.2 requires the exact qualified Plane integrator")
    if not isinstance(nonlinear, BerisEdwardsQNonlinearModel):
        raise TypeError("P5.2 requires the qualified Beris-Edwards Q RHS")
    if not isinstance(stokes, BerisEdwardsFreeSlipStokes):
        raise TypeError("P5.2 requires the qualified Beris-Edwards Stokes model")
    projector_owners = (
        model.spectral_projector,
        nonlinear.spectral_projector,
        stokes.spectral_projector,
        integrator.spectral_projector,
    )
    if any(owner is not projector for owner in projector_owners):
        raise ValueError("Plane operators must share one projector identity")
    if stokes.transform_backend is not solver.transform_backend:
        raise ValueError("Stokes and solver must share one transform backend")
    if nonlinear.pointwise_kernels is not stokes.pointwise_kernels:
        raise ValueError("Q and Stokes models must share pointwise kernels")
    if nonlinear.q_gradient_cache is not stokes.q_gradient_cache:
        raise ValueError("Q and Stokes models must share the Q-gradient cache")
    return PlaneCompiledOperatorBindings(
        prepare_algebraic=integrator._prepare_algebraic,
        explicit_rhs=integrator._explicit_rhs,
        project_dynamic_spectra=integrator._project_dynamic_spectra,
        inverse_dynamic_spectra=integrator._inverse_dynamic_spectra,
        refresh_dynamic_spectra=(
            integrator._refresh_dynamic_spectra_operation
        ),
        q_rhs_kernel=nonlinear,
        stokes_kernel=stokes,
        projector=projector,
    )


def bind_plane_compiled_v2(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    solver: object,
    projector: BasisAwareSpectralProjector,
) -> PlaneCompiledV2BindingPlan:
    """Audit and bind one untouched legacy Plane construction.

    The returned plan adopts references only.  It performs no timestep and
    allocates no persistent tensor or workspace.
    """

    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    if run_spec.runtime_path is not PlaneRuntimePath.LEGACY_PRODUCTION:
        raise ValueError("P5.2 binds only the legacy production oracle")
    required_solver_surface = (
        "batchsize",
        "dt",
        "dtype",
        "fields",
        "integrator",
        "L",
        "model",
        "parameters",
        "shape",
        "spectral_shape",
        "transform_backend",
    )
    missing_solver_surface = tuple(
        name for name in required_solver_surface if not hasattr(solver, name)
    )
    if missing_solver_surface:
        raise TypeError(
            "solver lacks the required Plane construction surface: "
            f"{missing_solver_surface!r}"
        )
    if not isinstance(projector, BasisAwareSpectralProjector):
        raise TypeError("projector must be a BasisAwareSpectralProjector")

    declaration = plane_compiled_v2_declaration()
    components = decompose_plane_beris_edwards_run_spec(run_spec)
    domain = components.geometry.domain
    numerics = components.numerics
    material = components.physics.material
    stokes_request = components.physics.stokes
    preset = components.preset
    time_stepping = components.time_stepping
    execution = components.execution
    expected_boundaries = components.effective_boundaries.to_legacy()

    expected_names = declaration.layout.component_order
    expected_indices = {
        name: index for index, name in enumerate(expected_names)
    }
    fields = solver.fields
    if fields.name_to_idx != expected_indices:
        raise ValueError("Plane field names or indices do not match declaration")
    if fields.dyn_count != 5 or fields.stat_count != 4:
        raise ValueError("Plane dynamic/static field counts are incompatible")
    if tuple(entry[0] for entry in solver.model.dyn_fields) != (
        declaration.layout.evolved_components
    ):
        raise ValueError("dynamic field registration order is incompatible")
    if tuple(entry[0] for entry in solver.model.stat_fields) != (
        declaration.layout.algebraic_components
    ):
        raise ValueError("algebraic field registration order is incompatible")

    if tuple(solver.shape) != domain.shape or tuple(solver.L) != domain.lengths:
        raise ValueError("solver domain does not match the resolved run spec")
    if solver.batchsize != 1 or fields.batchsize != 1:
        raise ValueError("compiled Plane binding currently requires batch size 1")
    if solver.dt != time_stepping.dt:
        raise ValueError("solver timestep does not match the resolved run spec")
    real_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[numerics.precision.value]
    spectral_dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[real_dtype]
    device = fields.spatial.device
    if solver.dtype != real_dtype or fields.dtype != real_dtype:
        raise ValueError("solver real dtype does not match the run spec")
    if solver.transform_backend.real_dtype != real_dtype:
        raise ValueError("transform backend real dtype is incompatible")
    if solver.transform_backend.spectral_dtype != spectral_dtype:
        raise ValueError("transform backend spectral dtype is incompatible")
    if projector.transform_backend is not solver.transform_backend:
        raise ValueError("projector must share the solver transform backend")
    if projector.rule != numerics.dealias_rule.value:
        raise ValueError("projector dealias rule does not match the run spec")
    if projector.transform_execution != (
        numerics.projected_transform_execution.value
    ):
        raise ValueError("projected transform execution is incompatible")
    backend = solver.transform_backend
    if (
        backend.execution_order != numerics.transform_execution_order.value
        or backend.spectral_storage != numerics.spectral_storage.value
        or backend.hermitian_axis != numerics.hermitian_axis
    ):
        raise ValueError("transform backend layout does not match the run spec")
    if (
        torch.device(backend.device) != device
        or torch.device(projector.device) != device
    ):
        raise ValueError("transform backend and projector device are incompatible")
    if projector.real_dtype != real_dtype or projector.shape != (
        solver.spectral_shape
    ):
        raise ValueError("projector dtype or spectral shape is incompatible")

    semantic_groups = tuple(
        _bind_transform_group(
            fields,
            name=group.name,
            components=group.components,
        )
        for group in declaration.layout.transform_groups
    )
    _validate_semantic_groups(declaration, semantic_groups)
    expected_group_boundaries = {
        "q_neumann": tuple(expected_boundaries["q"]),
        "tangential_velocity_neumann": tuple(
            expected_boundaries["tangential_velocity"]
        ),
        "normal_velocity_dirichlet": tuple(
            expected_boundaries["normal_velocity"]
        ),
        "pressure_modal_neumann": tuple(
            expected_boundaries["pressure_modal"]
        ),
    }
    for group in semantic_groups:
        if group.boundary_conditions != expected_group_boundaries[group.name]:
            raise ValueError(f"boundary mismatch for group {group.name}")

    dynamic_groups = tuple(
        tuple(group) for group in solver.integrator.dynamic_transform_groups
    )
    static_groups = tuple(
        tuple(group) for group in solver.model.static_transform_groups
    )
    if dynamic_groups != ((0, 1, 2, 3, 4),):
        raise ValueError("production dynamic transform grouping changed")
    if static_groups != ((5, 6, 8), (7,)):
        raise ValueError("production static transform grouping changed")
    execution_groups = (
        _bind_transform_group(
            fields,
            name="dynamic_q",
            components=tuple(expected_names[index] for index in dynamic_groups[0]),
        ),
        _bind_transform_group(
            fields,
            name="static_neumann",
            components=tuple(expected_names[index] for index in static_groups[0]),
        ),
        _bind_transform_group(
            fields,
            name="static_dirichlet",
            components=tuple(expected_names[index] for index in static_groups[1]),
        ),
    )

    physical_shape = (9, 1, *domain.shape)
    spectral_shape = (9, 1, *solver.spectral_shape)
    dynamic_spectral_shape = (5, 1, *solver.spectral_shape)
    spatial = _require_tensor(
        fields.spatial,
        name="fields.spatial",
        shape=physical_shape,
        dtype=real_dtype,
        device=device,
    )
    spectral = _require_tensor(
        fields.spectral,
        name="fields.spectral",
        shape=spectral_shape,
        dtype=spectral_dtype,
        device=device,
    )
    linear_operator = _require_tensor(
        fields.L_hat,
        name="fields.L_hat",
        shape=dynamic_spectral_shape,
        dtype=real_dtype,
        device=device,
    )
    integrator = solver.integrator
    denominator = _require_tensor(
        integrator.denom,
        name="integrator.denom",
        shape=dynamic_spectral_shape,
        dtype=real_dtype,
        device=device,
    )
    activity = _require_tensor(
        solver.parameters["alpha"],
        name="parameters.alpha",
        shape=(),
        dtype=real_dtype,
        device=device,
    )
    _require_distinct_storage(
        (
            ("fields.spatial", spatial),
            ("fields.spectral", spectral),
            ("fields.L_hat", linear_operator),
            ("integrator.denom", denominator),
            ("parameters.alpha", activity),
        )
    )
    expected_denominator = 1.0 - linear_operator * time_stepping.dt
    if not torch.equal(denominator, expected_denominator):
        raise ValueError("semi-implicit denominator does not match 1-dt*L")
    if float(activity.item()) != float(preset.zeta):
        raise ValueError("activity tensor does not match the resolved preset")

    state = integrator.runtime_state
    if not isinstance(state, RuntimeState):
        raise TypeError("qualified integrator lacks RuntimeState")
    if state.component_names != declaration.layout.evolved_components:
        raise ValueError("runtime state component ordering is incompatible")
    if (
        state.progress.completed_steps != 0
        or state.progress.refresh_step_count != 0
        or state.progress.refresh_count != 0
        or state.representations.current is not CurrentRepresentation.SYNCHRONIZED
        or state.representations.generation != 0
    ):
        raise ValueError("P5.2 requires an untouched construction-time state")
    if state.progress.dt != time_stepping.dt:
        raise ValueError("runtime progress timestep is incompatible")
    if state.progress.refresh_interval != (
        time_stepping.spectral_refresh.effective_interval_steps
    ):
        raise ValueError("spectral-refresh interval is incompatible")
    evolved_physical = _require_tensor(
        state.physical,
        name="state.physical",
        shape=(5, 1, *domain.shape),
        dtype=real_dtype,
        device=device,
    )
    evolved_spectral = _require_tensor(
        state.spectral,
        name="state.spectral",
        shape=dynamic_spectral_shape,
        dtype=spectral_dtype,
        device=device,
    )
    algebraic_physical = spatial[5:9]
    algebraic_spectral = spectral[5:9]
    _require_same_storage(
        evolved_physical,
        spatial,
        name="state.physical",
    )
    _require_same_storage(
        evolved_spectral,
        spectral,
        name="state.spectral",
    )
    _require_same_storage(
        algebraic_physical,
        spatial,
        name="algebraic physical fields",
    )
    _require_same_storage(
        algebraic_spectral,
        spectral,
        name="algebraic spectral fields",
    )

    step_metadata = integrator._step_program.to_metadata()
    declared_order = [stage.value for stage in declaration.operation_order]
    if step_metadata.get("scheme") != "projected_semi_implicit_euler":
        raise ValueError("legacy integrator scheme is incompatible")
    if step_metadata.get("operation_order") != declared_order:
        raise ValueError("legacy operation graph differs from declaration")
    if step_metadata.get("connected_runtime") is not None:
        raise ValueError("qualified step program connection metadata changed")

    workspace = integrator.runtime_workspace
    if not isinstance(workspace, RuntimeWorkspace):
        raise TypeError("qualified integrator lacks RuntimeWorkspace")
    if workspace.active or workspace.generation != 0:
        raise ValueError("P5.2 requires an untouched legacy workspace")
    legacy_workspace_bytes = workspace.plan.required_bytes
    if legacy_workspace_bytes != 0:
        raise RuntimeError(
            "legacy assembly retains workspace storage; introduce a shared "
            "assembly factory before compiled-v2 binding"
        )

    operators = _validate_operator_graph(solver, projector)
    nonlinear = operators.q_rhs_kernel
    stokes = operators.stokes_kernel
    allowed_kernel_dtypes = {torch.bool, real_dtype, spectral_dtype}
    for name, buffer in stokes.named_buffers():
        if buffer.device != device:
            raise ValueError(f"Stokes buffer {name} is on the wrong device")
        if buffer.dtype not in allowed_kernel_dtypes:
            raise ValueError(f"Stokes buffer {name} has an incompatible dtype")
        if buffer.dtype != torch.bool and not bool(
            torch.isfinite(buffer).all().item()
        ):
            raise ValueError(f"Stokes buffer {name} must be finite")
    scalar_bindings = (
        PlaneScalarBinding(
            "dt",
            _require_close(integrator.dt, time_stepping.dt, "dt"),
            "time_stepping.dt",
        ),
        PlaneScalarBinding(
            "ldg_a",
            _require_close(stokes.ldg_a, material.ldg_a, "ldg_a"),
            "material.ldg_a",
        ),
        PlaneScalarBinding(
            "ldg_b",
            _require_close(stokes.ldg_b, material.ldg_b, "ldg_b"),
            "material.ldg_b",
        ),
        PlaneScalarBinding(
            "ldg_c",
            _require_close(stokes.ldg_c, material.ldg_c, "ldg_c"),
            "material.ldg_c",
        ),
        PlaneScalarBinding(
            "ldg_l1",
            _require_close(stokes.ldg_l1, preset.ldg_l1, "ldg_l1"),
            "preset.ldg_l1",
        ),
        PlaneScalarBinding(
            "rotational_viscosity",
            _require_close(
                material.gamma,
                preset.rotational_viscosity,
                "rotational_viscosity",
            ),
            "preset.rotational_viscosity",
        ),
        PlaneScalarBinding(
            "flow_alignment",
            _require_close(
                stokes.flow_alignment,
                material.flow_alignment,
                "flow_alignment",
            ),
            "material.flow_alignment",
        ),
        PlaneScalarBinding(
            "viscosity",
            _require_close(
                stokes.viscosity,
                stokes_request.viscosity,
                "viscosity",
            ),
            "stokes.viscosity",
        ),
        PlaneScalarBinding(
            "friction",
            _require_close(
                stokes.friction,
                stokes_request.friction,
                "friction",
            ),
            "stokes.friction",
        ),
        PlaneScalarBinding(
            "beta",
            _require_close(stokes.beta, material.beta, "beta"),
            "material.beta",
        ),
        PlaneScalarBinding(
            "zeta",
            _require_close(activity.item(), preset.zeta, "zeta"),
            "preset.zeta",
        ),
    )
    if nonlinear.ldg_b_over_gamma != material.ldg_b / material.gamma:
        raise ValueError("Q RHS ldg_b/gamma coefficient is incompatible")
    if nonlinear.ldg_c_over_gamma != material.ldg_c / material.gamma:
        raise ValueError("Q RHS ldg_c/gamma coefficient is incompatible")
    if nonlinear.flow_alignment != material.flow_alignment:
        raise ValueError("Q RHS flow-alignment coefficient is incompatible")
    if stokes.zero_mode_policy != (
        stokes_request.tangential_zero_mode_policy.value
    ):
        raise ValueError("Stokes zero-mode policy is incompatible")
    if stokes.molecular_field_linear_space != (
        execution.molecular_field_linear_space
    ):
        raise ValueError("molecular-field execution policy is incompatible")
    if stokes.stress_divergence_sum_space != (
        execution.stress_divergence_sum_space
    ):
        raise ValueError("stress-divergence execution policy is incompatible")

    tensors = PlaneCompiledTensorBindings(
        evolved_physical=evolved_physical,
        algebraic_physical=algebraic_physical,
        evolved_spectral=evolved_spectral,
        algebraic_spectral=algebraic_spectral,
        linear_operator=linear_operator,
        denominator=denominator,
        activity=activity,
    )
    return PlaneCompiledV2BindingPlan(
        declaration=declaration,
        runtime_identity_sha256=run_spec.runtime_identity_sha256(),
        fields=fields,
        state=state,
        tensors=tensors,
        operators=operators,
        scalar_bindings=scalar_bindings,
        semantic_transform_groups=semantic_groups,
        execution_transform_groups=execution_groups,
        legacy_workspace_bytes=legacy_workspace_bytes,
        additional_workspace_bytes=0,
    )


__all__ = [
    "PlaneBoundTransformGroup",
    "PlaneCompiledOperatorBindings",
    "PlaneCompiledTensorBindings",
    "PlaneCompiledV2BindingPlan",
    "PlaneScalarBinding",
    "bind_plane_compiled_v2",
]
