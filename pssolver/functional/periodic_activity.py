"""Pure batch-one activity control for the qualified periodic runtime.

This P9.2--P9.3 runtime is deliberately narrow.  It reuses the qualified periodic
complete-stress models, transforms, direct Stokes solve, projector, and IMEX
coefficients while making every trajectory-dependent value an explicit tensor
in a flat state tuple.  It does not wrap or extend the legacy
control implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pssolver.applications.periodic_beris_edwards import (
    load_periodic_initial_q,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_PERIODIC_APPLICATION,
    compile_public_simulation,
)
from pssolver.configuration.simulation import SimulationSpec
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import (
    PERIODIC_BOUNDARIES,
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)

from .contracts import (
    FUNCTIONAL_API_VERSION,
    FunctionalCapabilitySet,
    FunctionalControlFieldSpec,
    FunctionalControls,
    FunctionalObservationSpec,
    FunctionalObservations,
    FunctionalRuntimeConstructionRequest,
    FunctionalRuntimeIdentity,
    FunctionalState,
    FunctionalStateSpec,
    FunctionalTensorSpec,
)
from .periodic_checkpoint import PeriodicActivityCheckpointBridge


_ACTIVITY_NAME = "activity"
_OBSERVATION_NAMES = ("Q", "velocity", "pressure")

# P9.5 qualified bitwise replay for this batch-one runtime on both CPU and the
# allocated NVIDIA H100 PCIe under one fixed dtype and execution identity.  Do
# not weaken this declaration merely because the allocated device is CUDA: an
# independent checkpointing consumer must be able to reject a runtime whose
# replay capability is genuinely unqualified.
PERIODIC_ACTIVITY_DETERMINISTIC_REPLAY = "bitwise"


def _allocated_device(value: object) -> str:
    try:
        requested = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("functional execution device is invalid") from exc
    if requested.type != "cuda":
        return str(requested)
    if not torch.cuda.is_available():
        raise RuntimeError("functional CUDA construction requires CUDA")
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("functional CUDA device index is unavailable")
    return str(torch.device("cuda", index))


def _spectral_dtype(real_dtype: str) -> str:
    return {"float32": "complex64", "float64": "complex128"}[real_dtype]


def _functional_declarations(
    simulation: SimulationSpec,
    *,
    device: str,
) -> tuple[
    FunctionalStateSpec,
    tuple[FunctionalControlFieldSpec, ...],
    tuple[FunctionalObservationSpec, ...],
]:
    shape = tuple(simulation.geometry.domain.shape)
    precision = simulation.numerics.precision.value
    storage = simulation.numerics.spectral_storage.value
    spectral_shape = list(shape)
    if storage == "hermitian_half":
        axis = simulation.numerics.hermitian_axis
        if axis is None:
            raise ValueError("Hermitian functional state requires hermitian_axis")
        spectral_shape[axis] = spectral_shape[axis] // 2 + 1
    spectral_shape = tuple(spectral_shape)

    q_physical = FunctionalTensorSpec(
        name="q_physical",
        shape=(len(Q_COMPONENTS), 1, *shape),
        dtype=precision,
        device=device,
        batch_axis=1,
        layout="component_batch_xyz",
        meaning=(
            "physical five-component symmetric-traceless Q state at the "
            "current timestep"
        ),
        component_names=Q_COMPONENTS,
    )
    q_spectral = FunctionalTensorSpec(
        name="q_spectral",
        shape=(len(Q_COMPONENTS), 1, *spectral_shape),
        dtype=_spectral_dtype(precision),
        device=device,
        batch_axis=1,
        layout="component_batch_native_spectral_xyz",
        meaning=(
            "native projected Q spectrum retained as persistent IMEX state"
        ),
        component_names=Q_COMPONENTS,
    )
    activity_tensor = FunctionalTensorSpec(
        name=_ACTIVITY_NAME,
        shape=(1, *shape),
        dtype=precision,
        device=device,
        batch_axis=0,
        layout="batch_xyz",
        meaning="cell-centered activity coefficient alpha",
    )
    dealias = simulation.numerics.dealias_rule.value
    projected_execution = (
        simulation.numerics.projected_transform_execution.value
    )
    control = FunctionalControlFieldSpec(
        name=_ACTIVITY_NAME,
        tensor=activity_tensor,
        equation_term="div(beta * alpha * Q)",
        injection_order="form_beta_alpha_q_product_before_divergence",
        dealiasing_identity=(
            f"{dealias}:{projected_execution}:"
            "project_product_then_spectral_divergence"
        ),
        grid_location="cell_centered_physical_grid",
        broadcast_rules=("exact_batch_and_spatial_shape",),
        admissible_min=0.0,
    )
    q_observation = FunctionalObservationSpec(
        name="Q",
        tensor=FunctionalTensorSpec(
            name="Q",
            shape=q_physical.shape,
            dtype=precision,
            device=device,
            batch_axis=1,
            layout=q_physical.layout,
            meaning="physical Q tensor at the input state",
            component_names=Q_COMPONENTS,
        ),
        convention="input physical compact Q",
        control_dependent=False,
        terminal_available_without_control=True,
    )
    velocity_observation = FunctionalObservationSpec(
        name="velocity",
        tensor=FunctionalTensorSpec(
            name="velocity",
            shape=(3, 1, *shape),
            dtype=precision,
            device=device,
            batch_axis=1,
            layout="component_batch_xyz",
            meaning="physical incompressible velocity at the input state",
            component_names=("ux", "uy", "uz"),
        ),
        convention="direct periodic Stokes velocity under current activity",
        control_dependent=True,
        terminal_available_without_control=False,
    )
    pressure_observation = FunctionalObservationSpec(
        name="pressure",
        tensor=FunctionalTensorSpec(
            name="pressure",
            shape=(1, *shape),
            dtype=precision,
            device=device,
            batch_axis=0,
            layout="batch_xyz",
            meaning="zero-mean pressure at the input state",
        ),
        convention="direct periodic Stokes zero-mean pressure",
        control_dependent=True,
        terminal_available_without_control=False,
    )
    return (
        FunctionalStateSpec((q_physical, q_spectral)),
        (control,),
        (q_observation, velocity_observation, pressure_observation),
    )


def periodic_activity_functional_request(
    simulation: SimulationSpec,
) -> FunctionalRuntimeConstructionRequest:
    """Return the exact P9.2 declaration for one qualified simulation."""

    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    compiled = compile_public_simulation(simulation)
    if compiled.application != PUBLIC_PERIODIC_APPLICATION:
        raise ValueError(
            "P9.2 functional execution requires the qualified periodic "
            "complete-stress application"
        )
    execution = simulation.execution.options
    if execution["pointwise_execution"] != "eager":
        raise ValueError("P9.2 functional execution requires eager pointwise kernels")
    device = _allocated_device(execution["device"])
    _, controls, observations = _functional_declarations(
        simulation,
        device=device,
    )
    return FunctionalRuntimeConstructionRequest(
        simulation=simulation,
        control_fields=controls,
        observations=observations,
        batch_size=1,
    )


class PeriodicActivityFunctionalRuntime:
    """Pure explicit-state view of the qualified periodic production map."""

    def __init__(
        self,
        request: FunctionalRuntimeConstructionRequest,
        *,
        adapter,
        snapshot_sha256: str,
        production_runtime_identity_sha256: str,
    ) -> None:
        if not isinstance(request, FunctionalRuntimeConstructionRequest):
            raise TypeError("request must be FunctionalRuntimeConstructionRequest")
        device = request.control_fields[0].tensor.device
        state_spec, controls, observations = _functional_declarations(
            request.simulation,
            device=device,
        )
        if tuple(value.to_metadata() for value in request.control_fields) != tuple(
            value.to_metadata() for value in controls
        ):
            raise ValueError("functional activity-control declaration is not canonical")
        if tuple(value.to_metadata() for value in request.observations) != tuple(
            value.to_metadata() for value in observations
        ):
            raise ValueError("functional observation declaration is not canonical")

        self._request = request
        self._adapter = adapter
        self._solver = adapter.solver
        self._projector = adapter.projector
        self._state_spec = state_spec
        self._control_specs = controls
        self._observation_specs = observations
        replay_capability = PERIODIC_ACTIVITY_DETERMINISTIC_REPLAY
        self._capabilities = FunctionalCapabilitySet(
            supported_batch_sizes=(1,),
            pure_step=True,
            combined_step_and_observe=True,
            deterministic_replay=replay_capability,
            differentiability="torch_autograd",
            durable_checkpoint_bridge=True,
            explicit_jvp=False,
            explicit_vjp=False,
            inner_solve_gradient="direct_periodic_fourier_autograd",
            differentiable_inputs=("state", "activity"),
        )

        # A functional call must not depend on a previously staged gradient.
        # The immutable transform/operator caches remain implementation detail;
        # all trajectory-dependent arrays are supplied in FunctionalState.
        self._solver.model.nlmodel.q_gradient_cache = None
        self._solver.model.static_model.q_gradient_cache = None
        self._solver.model.static_model.pressure_diagnostics = False

        fields = self._solver.fields
        dynamic = fields.dyn_count
        self._initial_state = (
            fields.spatial[:dynamic].detach().clone(),
            fields.spectral[:dynamic].detach().clone(),
        )
        identities = request.simulation.identity_metadata()
        execution_identity = dict(identities["execution"])
        execution_identity["functional_runtime"] = {
            "kind": "periodic_activity_batch_one",
            "device": device,
            "pointwise_execution": "eager",
            "q_gradient_reuse": False,
            "pressure_solver": "direct_periodic_fourier",
            "spectral_refresh": "disabled",
            "activity_product_before_divergence": True,
            "snapshot_sha256": snapshot_sha256,
            "fallback_allowed": False,
            "fallback_used": False,
            "deterministic_replay": replay_capability,
            "durable_checkpoint_bridge_version": 1,
        }
        self._identity = FunctionalRuntimeIdentity(
            scientific=dict(identities["scientific"]),
            discretization=dict(identities["discretization"]),
            execution=execution_identity,
            state_layout=state_spec.to_metadata(),
        )
        self._checkpoint_bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=self._identity,
            production_runtime_identity_sha256=(
                production_runtime_identity_sha256
            ),
            backend_restart=adapter.backend_restart_metadata(),
        )

    @property
    def api_version(self) -> str:
        return FUNCTIONAL_API_VERSION

    @property
    def state_spec(self) -> FunctionalStateSpec:
        return self._state_spec

    @property
    def control_specs(self) -> tuple[FunctionalControlFieldSpec, ...]:
        return self._control_specs

    @property
    def control_field_schema(self) -> tuple[FunctionalControlFieldSpec, ...]:
        return self._control_specs

    @property
    def observation_specs(self) -> tuple[FunctionalObservationSpec, ...]:
        return self._observation_specs

    @property
    def capabilities(self) -> FunctionalCapabilitySet:
        return self._capabilities

    @property
    def checkpoint_bridge(self) -> PeriodicActivityCheckpointBridge:
        return self._checkpoint_bridge

    def identity(self) -> FunctionalRuntimeIdentity:
        return self._identity

    def initial_state(self) -> FunctionalState:
        return tuple(value.clone() for value in self._initial_state)

    @staticmethod
    def _validate_step_index(step_index: int) -> None:
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError("step_index must be an integer")
        if step_index < 0:
            raise ValueError("step_index must be non-negative")

    def _validate_state(self, state: FunctionalState) -> None:
        self._state_spec.validate(state)
        if any(not bool(torch.isfinite(value).all()) for value in state):
            raise ValueError("functional state contains NaN or Inf")

    def _activity(self, controls: FunctionalControls) -> torch.Tensor:
        if not isinstance(controls, Mapping):
            raise TypeError("functional controls must be a mapping")
        if set(controls) != {_ACTIVITY_NAME}:
            raise ValueError("functional controls must contain only activity")
        value = controls[_ACTIVITY_NAME]
        self._control_specs[0].validate(value)
        return value

    def _bind_input_state(
        self,
        state: FunctionalState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_physical, q_spectral = state
        fields = self._solver.fields
        static_count = fields.stat_count
        spatial_zeros = q_physical.new_zeros(
            (static_count, q_physical.shape[1], *q_physical.shape[2:])
        )
        spectral_zeros = q_spectral.new_zeros(
            (static_count, q_spectral.shape[1], *q_spectral.shape[2:])
        )
        # Concatenation creates fresh owned storage: neither the caller's
        # physical nor spectral tensor is mutated by model execution.
        fields.spatial = torch.cat((q_physical, spatial_zeros), dim=0)
        fields.spectral = torch.cat((q_spectral, spectral_zeros), dim=0)
        return q_physical, q_spectral

    def _flow_and_rhs(
        self,
        state: FunctionalState,
        activity: torch.Tensor,
        *,
        need_rhs: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        q_physical, q_spectral = self._bind_input_state(state)
        model = self._solver.model
        fields = self._solver.fields
        # Controls are call arguments, not mutable entries in the production
        # parameter registry.  The existing model adapters accept this minimal
        # per-call mapping without changing their equation implementation.
        static_hat = model.static_model(fields, {"alpha": activity})
        static_physical = self._projector.inverse_transform(
            static_hat,
            PERIODIC_BOUNDARIES,
        )
        fields.spatial = torch.cat((q_physical, static_physical), dim=0)
        fields.spectral = torch.cat((q_spectral, static_hat), dim=0)
        observations = {
            "Q": q_physical,
            "velocity": static_physical[:3],
            "pressure": static_physical[3],
        }
        rhs = model.nlmodel(fields, {}) if need_rhs else None
        return observations, rhs

    def _next_state(
        self,
        state: FunctionalState,
        rhs: torch.Tensor,
    ) -> FunctionalState:
        q_spectral = state[1]
        integrator = self._solver.integrator
        next_spectral = (
            q_spectral + integrator.dt * rhs
        ) / integrator.denom
        next_spectral = self._projector.project(
            next_spectral,
            PERIODIC_BOUNDARIES,
        )
        next_physical = self._projector.inverse_transform(
            next_spectral,
            PERIODIC_BOUNDARIES,
        )
        return (next_physical, next_spectral)

    def step(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> FunctionalState:
        self._validate_step_index(step_index)
        self._validate_state(state)
        activity = self._activity(controls)
        _, rhs = self._flow_and_rhs(state, activity, need_rhs=True)
        if rhs is None:  # pragma: no cover - guarded by need_rhs=True
            raise AssertionError("functional RHS was not computed")
        return self._next_state(state, rhs)

    def observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls | None,
    ) -> FunctionalObservations:
        self._validate_state(state)
        if controls is None:
            return {"Q": state[0]}
        activity = self._activity(controls)
        observations, _ = self._flow_and_rhs(
            state,
            activity,
            need_rhs=False,
        )
        return observations

    def step_and_observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> tuple[FunctionalState, FunctionalObservations]:
        self._validate_step_index(step_index)
        self._validate_state(state)
        activity = self._activity(controls)
        observations, rhs = self._flow_and_rhs(
            state,
            activity,
            need_rhs=True,
        )
        if rhs is None:  # pragma: no cover - guarded by need_rhs=True
            raise AssertionError("functional RHS was not computed")
        return self._next_state(state, rhs), observations


class PeriodicActivityFunctionalRuntimeFactory:
    """Fail-closed factory for the single P9.2 model--geometry pair."""

    def build(
        self,
        request: FunctionalRuntimeConstructionRequest,
    ) -> PeriodicActivityFunctionalRuntime:
        if not isinstance(request, FunctionalRuntimeConstructionRequest):
            raise TypeError("request must be FunctionalRuntimeConstructionRequest")
        expected = periodic_activity_functional_request(request.simulation)
        if request.to_metadata() != expected.to_metadata():
            raise ValueError("functional construction request is not canonical")
        compiled = compile_public_simulation(request.simulation)
        run_spec = compiled.run_spec
        initial_values, _, snapshot_sha256 = load_periodic_initial_q(run_spec)
        adapter = build_periodic_beris_edwards_runtime(
            PeriodicRuntimeBuildRequest(
                run_spec=run_spec,
                initial_values=initial_values,
                device=request.control_fields[0].tensor.device,
            )
        )
        return PeriodicActivityFunctionalRuntime(
            request,
            adapter=adapter,
            snapshot_sha256=snapshot_sha256,
            production_runtime_identity_sha256=(
                run_spec.runtime_identity_sha256()
            ),
        )


def build_functional_runtime(
    request: FunctionalRuntimeConstructionRequest,
) -> PeriodicActivityFunctionalRuntime:
    """Build a qualified functional runtime or reject before execution."""

    return PeriodicActivityFunctionalRuntimeFactory().build(request)


__all__ = [
    "PeriodicActivityFunctionalRuntime",
    "PeriodicActivityFunctionalRuntimeFactory",
    "build_functional_runtime",
    "periodic_activity_functional_request",
]
