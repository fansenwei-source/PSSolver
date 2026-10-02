"""Pure batch-one activity control for the complete-stress Channel runtime.

P9.7.4 reuses the qualified production transforms, force model, projected
Euler coefficients, and modal Stokes operators.  It deliberately bypasses the
production pressure warm start and routes every pressure solve through the
P9.7.3 implicit-adjoint node with a canonical zero initial guess.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pssolver.applications.channel_beris_edwards import load_channel_initial_q
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_COMPLETE_APPLICATION,
    compile_public_simulation,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
)
from pssolver.models.active_nematics import CHANNEL_Q_BOUNDARY_CONDITIONS
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)

from .channel_activity import (
    CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
    _state_control_observation_specs,
    channel_activity_functional_request,
)
from .channel_checkpoint import ChannelActivityCheckpointBridge
from .channel_pressure_adjoint import ChannelImplicitPressureAdjoint
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
)


CHANNEL_ACTIVITY_RUNTIME_STAGE = "P9.7.4"
CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY = "bitwise"
_ACTIVITY_NAME = "activity"


class ChannelActivityFunctionalRuntime:
    """Explicit-state Channel step with a graph-bounded pressure adjoint."""

    def __init__(
        self,
        request: FunctionalRuntimeConstructionRequest,
        *,
        adapter,
        snapshot_sha256: str,
        production_runtime_identity_sha256: str,
        production_backend_restart: Mapping[str, object],
    ) -> None:
        if not isinstance(request, FunctionalRuntimeConstructionRequest):
            raise TypeError("request must be a FunctionalRuntimeConstructionRequest")
        device = request.control_fields[0].tensor.device
        state_spec, controls, observations = _state_control_observation_specs(
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
        self._flow_model = self._solver.model.static_model
        self._pressure = ChannelImplicitPressureAdjoint(self._flow_model)
        self._capabilities = FunctionalCapabilitySet(
            supported_batch_sizes=(1,),
            pure_step=True,
            combined_step_and_observe=True,
            deterministic_replay=CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY,
            differentiability="validated_custom_adjoint",
            durable_checkpoint_bridge=True,
            explicit_jvp=False,
            explicit_vjp=False,
            inner_solve_gradient="custom_implicit_pressure_adjoint_v1",
            differentiable_inputs=("state", "activity"),
        )

        # The functional map owns no trajectory-dependent cache.  These are
        # private objects built solely for this runtime; production defaults
        # and independently constructed production adapters remain unchanged.
        self._solver.model.nlmodel.q_gradient_cache = None
        self._flow_model.q_gradient_cache = None
        self._flow_model.pressure_guess = None

        fields = self._solver.fields
        dynamic = fields.dyn_count
        self._initial_state = (
            fields.spatial[:dynamic].detach().clone(),
            fields.spectral[:dynamic].detach().clone(),
        )
        identities = request.simulation.identity_metadata()
        execution_identity = dict(identities["execution"])
        pressure = dict(
            request.simulation.discretization_parameters["pressure_solver"]
        )
        execution_identity["functional_runtime"] = {
            "kind": CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
            "stage": CHANNEL_ACTIVITY_RUNTIME_STAGE,
            "device": device,
            "pointwise_execution": "eager",
            "q_gradient_reuse": False,
            "spectral_refresh": "disabled",
            "activity_product_before_divergence": True,
            "pressure_solver": {
                "kind": "channel_no_slip_modal_stokes_pcg",
                "algorithm": pressure["algorithm"],
                "relative_tolerance": pressure["relative_tolerance"],
                "max_iterations": pressure["max_iterations"],
                "fixed_iterations": pressure["fixed_iterations"],
                "stopping_precedence": (
                    "fixed_iterations_overrides_max_iterations;"
                    "relative_tolerance_is_early_exit_only"
                    if pressure["fixed_iterations"] is not None
                    else (
                        "relative_tolerance_required_with_max_iterations_cap"
                    )
                ),
                "gauge": "zero_mean",
                "initial_guess": "zero_every_call",
                "production_warm_start_read": False,
                "production_warm_start_written": False,
                "gradient": "custom_implicit_pressure_adjoint_v1",
            },
            "boundary_spaces": {
                "Q": list(CHANNEL_Q_BOUNDARY_CONDITIONS),
                "velocity": list(CHANNEL_VELOCITY_BOUNDARY_CONDITIONS),
                "pressure": list(CHANNEL_PRESSURE_BOUNDARY_CONDITIONS),
            },
            "snapshot_sha256": snapshot_sha256,
            "fallback_allowed": False,
            "fallback_used": False,
            "deterministic_replay": CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY,
            "durable_checkpoint_bridge_version": 1,
            "checkpoint_export": "functional_q_state_only",
            "production_checkpoint_import": "q_state_only_after_identity_gate",
        }
        self._identity = FunctionalRuntimeIdentity(
            scientific=dict(identities["scientific"]),
            discretization=dict(identities["discretization"]),
            execution=execution_identity,
            state_layout=state_spec.to_metadata(),
        )
        self._checkpoint_bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=self._identity,
            production_runtime_identity_sha256=(
                production_runtime_identity_sha256
            ),
            production_backend_restart=production_backend_restart,
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
    def checkpoint_bridge(self) -> ChannelActivityCheckpointBridge:
        return self._checkpoint_bridge

    def identity(self) -> FunctionalRuntimeIdentity:
        return self._identity

    def diagnostics(self) -> dict[str, object]:
        """Return stable outcomes from the latest primal/transpose solves."""

        return {
            "schema_version": 1,
            "runtime_kind": CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
            "pressure": self._pressure.pressure_solve_metadata(),
        }

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
        if any(not bool(value.detach().isfinite().all()) for value in state):
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
        fields.spatial = torch.cat((q_physical, spatial_zeros), dim=0)
        fields.spectral = torch.cat((q_spectral, spectral_zeros), dim=0)
        return q_physical, q_spectral

    def _solve_flow(
        self,
        activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = self._solver.fields
        force = self._flow_model.compute_nematic_force(fields, activity)
        force_hat = self._projector.forward_transform(
            force,
            CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
        )
        *velocity_hats, pressure_hat = self._pressure.solve_force_hats(
            *force_hat
        )
        velocity_hat = torch.stack(tuple(velocity_hats))
        velocity = self._projector.inverse_transform_full(
            velocity_hat,
            CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
        )
        pressure = self._projector.inverse_transform_full(
            pressure_hat,
            CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
        )
        return velocity, pressure, velocity_hat, pressure_hat

    def _flow_and_rhs(
        self,
        state: FunctionalState,
        activity: torch.Tensor,
        *,
        need_rhs: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        q_physical, q_spectral = self._bind_input_state(state)
        velocity, pressure, velocity_hat, pressure_hat = self._solve_flow(activity)
        fields = self._solver.fields
        static_physical = torch.cat((velocity, pressure.unsqueeze(0)), dim=0)
        static_spectral = torch.cat(
            (velocity_hat, pressure_hat.unsqueeze(0)),
            dim=0,
        )
        fields.spatial = torch.cat((q_physical, static_physical), dim=0)
        fields.spectral = torch.cat((q_spectral, static_spectral), dim=0)
        observations = {
            "Q": q_physical,
            "velocity": velocity,
            "pressure": pressure,
        }
        rhs = self._solver.model.nlmodel(fields, {}) if need_rhs else None
        return observations, rhs

    def _next_state(
        self,
        state: FunctionalState,
        rhs: torch.Tensor,
    ) -> FunctionalState:
        integrator = self._solver.integrator
        next_spectral = (
            state[1] + integrator.dt * rhs
        ) / integrator.denom
        next_spectral = self._projector.project_real_spectrum(
            next_spectral,
            CHANNEL_Q_BOUNDARY_CONDITIONS,
        )
        next_physical = self._projector.inverse_transform(
            next_spectral,
            CHANNEL_Q_BOUNDARY_CONDITIONS,
        )
        return next_physical, next_spectral

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
        if rhs is None:  # pragma: no cover
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
        observations, _ = self._flow_and_rhs(
            state,
            self._activity(controls),
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
        observations, rhs = self._flow_and_rhs(
            state,
            self._activity(controls),
            need_rhs=True,
        )
        if rhs is None:  # pragma: no cover
            raise AssertionError("functional RHS was not computed")
        return self._next_state(state, rhs), observations


class ChannelActivityFunctionalRuntimeFactory:
    """Fail-closed factory for the first P9.7 Channel combination."""

    def build(
        self,
        request: FunctionalRuntimeConstructionRequest,
    ) -> ChannelActivityFunctionalRuntime:
        if not isinstance(request, FunctionalRuntimeConstructionRequest):
            raise TypeError("request must be a FunctionalRuntimeConstructionRequest")
        expected = channel_activity_functional_request(request.simulation)
        if request.to_metadata() != expected.to_metadata():
            raise ValueError("functional construction request is not canonical")
        compiled = compile_public_simulation(request.simulation)
        if compiled.application != PUBLIC_CHANNEL_COMPLETE_APPLICATION:
            raise ValueError("P9.7.4 requires the complete-stress Channel application")
        run_spec = compiled.run_spec
        initial_values, _, snapshot_sha256 = load_channel_initial_q(run_spec)
        adapter = build_channel_beris_edwards_runtime(
            ChannelBerisEdwardsRuntimeBuildRequest(
                run_spec=run_spec,
                initial_values=initial_values,
                device=request.control_fields[0].tensor.device,
            )
        )
        production_backend_restart = adapter.backend_restart_metadata()
        return ChannelActivityFunctionalRuntime(
            request,
            adapter=adapter,
            snapshot_sha256=snapshot_sha256,
            production_runtime_identity_sha256=(
                run_spec.runtime_identity_sha256()
            ),
            production_backend_restart=production_backend_restart,
        )


def build_channel_activity_functional_runtime(
    request: FunctionalRuntimeConstructionRequest,
) -> ChannelActivityFunctionalRuntime:
    """Build the provisional P9.7.4 Channel runtime or fail closed."""

    return ChannelActivityFunctionalRuntimeFactory().build(request)


__all__ = [
    "CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY",
    "CHANNEL_ACTIVITY_RUNTIME_STAGE",
    "ChannelActivityFunctionalRuntime",
    "ChannelActivityFunctionalRuntimeFactory",
    "build_channel_activity_functional_runtime",
]
