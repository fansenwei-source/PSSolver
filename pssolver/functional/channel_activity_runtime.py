"""Functional 1.1 Channel activity on the Engine's zero-start pressure path."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from pssolver.configuration.execution_device import (
    configure_tf32_execution,
    scoped_tf32_execution,
)
from pssolver.configuration.legacy_channel import (
    channel_forward_dynamics_identity,
    load_channel_initial_q,
    translate_channel,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_COMPLETE_APPLICATION,
    compile_public_simulation,
)
from pssolver.engine.engine import Engine
from pssolver.engine.state import State
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
)
from pssolver.models.active_nematics import CHANNEL_Q_BOUNDARY_CONDITIONS

from .channel_activity import (
    CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
    _state_control_observation_specs,
    channel_activity_functional_capabilities,
    channel_activity_functional_request,
    channel_functional_pressure_identity,
)
from .channel_checkpoint import ChannelActivityCheckpointBridge
from .contracts import (
    FUNCTIONAL_API_VERSION,
    FunctionalCapabilitySet,
    FunctionalControlFieldSpec,
    FunctionalControls,
    FunctionalObservations,
    FunctionalObservationSpec,
    FunctionalRuntimeConstructionRequest,
    FunctionalRuntimeIdentity,
    FunctionalState,
    FunctionalStateSpec,
)
from .errors import FunctionalConvergenceError
from .pressure_metadata import (
    CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    channel_pressure_solve_diagnostics,
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
        engine,
        initial_state,
        snapshot_sha256: str,
        production_runtime_identity_sha256: str,
        production_backend_restart: Mapping[str, object],
        production_forward_dynamics: Mapping[str, object] | None = None,
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
        self._engine = engine
        self._state_spec = state_spec
        self._control_specs = controls
        self._observation_specs = observations
        self._tf32_requested = request.simulation.execution.options["tf32"]
        self._tf32_real_dtype = getattr(
            torch,
            request.simulation.numerics.precision.value,
        )
        self._tf32_device = device
        self._pressure = engine.constraint.functional_pressure_diagnostics(
            make_diagnostics=channel_pressure_solve_diagnostics,
            error=FunctionalConvergenceError,
            schema_version=CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
        )
        self._capabilities = channel_activity_functional_capabilities()

        dynamic = engine.evolved_count
        self._initial_state = (
            initial_state.spatial[:dynamic].detach().clone(),
            initial_state.spectral[:dynamic].detach().clone(),
        )
        identities = request.simulation.identity_metadata()
        execution_identity = dict(identities["execution"])
        execution_identity["functional_runtime"] = {
            "kind": CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
            "stage": CHANNEL_ACTIVITY_RUNTIME_STAGE,
            "device": device,
            "pointwise_execution": "eager",
            "q_gradient_reuse": False,
            "spectral_refresh": "disabled",
            "activity_product_before_divergence": True,
            "pressure_solver": channel_functional_pressure_identity(
                request.simulation
            ),
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
            "tf32_policy": configure_tf32_execution(
                self._tf32_requested,
                real_dtype=self._tf32_real_dtype,
                device=self._tf32_device,
            ),
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
            production_forward_dynamics=production_forward_dynamics,
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
            "schema_version": 2,
            "runtime_kind": CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
            "pressure": self._pressure.pressure_solve_metadata(),
        }

    def initial_state(self) -> FunctionalState:
        return tuple(value.clone() for value in self._initial_state)

    def execution_scope(self):
        """Own TF32 policy across an explicit forward/backward region."""

        return scoped_tf32_execution(
            self._tf32_requested,
            real_dtype=self._tf32_real_dtype,
            device=self._tf32_device,
        )

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

    def _engine_input(self, state, step_index):
        q_physical, q_spectral = state
        engine = self._engine
        static_count = len(engine.components) - engine.evolved_count
        spatial_zeros = q_physical.new_zeros(
            (static_count, q_physical.shape[1], *q_physical.shape[2:])
        )
        spectral_zeros = q_spectral.new_zeros(
            (static_count, q_spectral.shape[1], *q_spectral.shape[2:])
        )
        return State(
            torch.cat((q_spectral, spectral_zeros), dim=0),
            torch.cat((q_physical, spatial_zeros), dim=0),
            (),
            engine.constraint.seed(None),
            step_index,
            (step_index, 0),
            False,
            pure_evolved=state,
        )

    def _prepare_observation(self, state, controls, step_index):
        current = self._engine.prepare_constraints(
            self._engine_input(state, step_index),
            {"alpha": self._activity(controls)},
            diagnostics=self._pressure,
        )
        observations = self._engine.observe(current)
        observations["Q"] = state[0]
        observations["pressure"] = observations["pressure"][0]
        return current, observations

    def step(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> FunctionalState:
        with scoped_tf32_execution(
            self._tf32_requested,
            real_dtype=self._tf32_real_dtype,
            device=self._tf32_device,
        ):
            self._validate_step_index(step_index)
            self._validate_state(state)
            activity = self._activity(controls)
            result = self._engine.step(
                self._engine_input(state, step_index),
                {"alpha": activity},
                diagnostics=self._pressure,
            )
            return result.pure_evolved

    def observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls | None,
    ) -> FunctionalObservations:
        with scoped_tf32_execution(
            self._tf32_requested,
            real_dtype=self._tf32_real_dtype,
            device=self._tf32_device,
        ):
            self._validate_state(state)
            if controls is None:
                return {"Q": state[0]}
            _, observations = self._prepare_observation(state, controls, 0)
            return observations

    def step_and_observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> tuple[FunctionalState, FunctionalObservations]:
        with scoped_tf32_execution(
            self._tf32_requested,
            real_dtype=self._tf32_real_dtype,
            device=self._tf32_device,
        ):
            self._validate_step_index(step_index)
            self._validate_state(state)
            current, observations = self._prepare_observation(state, controls, step_index)
            result = self._engine.step(
                current, {"alpha": controls[_ACTIVITY_NAME]}, diagnostics=self._pressure
            )
            return result.pure_evolved, observations


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
        device = request.control_fields[0].tensor.device
        dtype = getattr(torch, request.simulation.numerics.precision.value)
        engine = Engine(
            translate_channel(run_spec, flavour="functional_1_1"), device=device, dtype=dtype
        )
        with scoped_tf32_execution(
            request.simulation.execution.options["tf32"], real_dtype=dtype, device=device
        ):
            initial_state = engine.initial_state(
                {name: value.to(device=device) for name, value in initial_values.items()}
            )
        production_backend_restart = {
            "kind": "channel_complete_stress_pressure_pcg",
            "state_keys": ["pressure_guess"],
            "reconstructed_state_keys": (
                [] if request.simulation.execution.options["disable_q_gradient_reuse"]
                else ["q_gradient_cache"]
            ),
        }

        return ChannelActivityFunctionalRuntime(
            request,
            engine=engine,
            initial_state=initial_state,
            snapshot_sha256=snapshot_sha256,
            production_runtime_identity_sha256=(
                run_spec.runtime_identity_sha256()
            ),
            production_backend_restart=production_backend_restart,
            production_forward_dynamics=(
                channel_forward_dynamics_identity(run_spec).to_metadata()
            ),
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
