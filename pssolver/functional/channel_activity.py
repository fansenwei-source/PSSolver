"""Qualified declarations for batch-one complete-stress Channel control.

Declaration remains side-effect free, while its capability and pressure
metadata describe the executable P9.7.4 runtime now available from the public
functional factory.
"""

from __future__ import annotations

import torch

from pssolver.configuration.channel_active_nematics_declarations import (
    channel_pressure_policy_metadata,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_COMPLETE_APPLICATION,
    compile_public_simulation,
)
from pssolver.configuration.simulation import SimulationSpec
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
)
from pssolver.models.active_nematics import (
    CHANNEL_Q_BOUNDARY_CONDITIONS,
    Q_COMPONENTS,
)

from .contracts import (
    FunctionalCapabilitySet,
    FunctionalControlFieldSpec,
    FunctionalObservationSpec,
    FunctionalRuntimeConstructionRequest,
    FunctionalRuntimeDeclaration,
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
)
from .channel_pressure import CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION
from .channel_pressure_adjoint import (
    CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION,
)
from .pressure_metadata import CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION


CHANNEL_ACTIVITY_FUNCTIONAL_KIND = "channel_activity_batch_one"
CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT = (
    "custom_implicit_pressure_adjoint_v2"
)
CHANNEL_FUNCTIONAL_PRESSURE_WARM_START = "disabled_in_functional_path"
_ACTIVITY_NAME = "activity"


def channel_activity_functional_capabilities() -> FunctionalCapabilitySet:
    """Return the one truthful declaration/runtime capability set."""

    return FunctionalCapabilitySet(
        supported_batch_sizes=(1,),
        pure_step=True,
        combined_step_and_observe=True,
        deterministic_replay="bitwise",
        differentiability="validated_custom_adjoint",
        durable_checkpoint_bridge=True,
        explicit_jvp=False,
        explicit_vjp=False,
        inner_solve_gradient=CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT,
        differentiable_inputs=("state", "activity"),
    )


def channel_functional_pressure_identity(
    simulation: SimulationSpec,
) -> dict[str, object]:
    """Build the canonical functional pressure identity payload."""

    pressure = dict(simulation.discretization_parameters["pressure_solver"])
    return {
        "kind": "channel_no_slip_modal_stokes_pcg",
        "algorithm": pressure["algorithm"],
        "relative_tolerance": pressure["relative_tolerance"],
        "max_iterations": pressure["max_iterations"],
        "fixed_iterations": pressure["fixed_iterations"],
        "derived_policy": channel_pressure_policy_metadata(
            relative_tolerance=pressure["relative_tolerance"],
            max_iterations=pressure["max_iterations"],
            fixed_iterations=pressure["fixed_iterations"],
            initial_guess_policy="zero_every_call",
        ),
        "diagnostics_schema_version": (
            CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION
        ),
        "gauge": "zero_mean",
        "production_warm_start": pressure["warm_start"],
        "functional_warm_start": False,
        "functional_warm_start_policy": CHANNEL_FUNCTIONAL_PRESSURE_WARM_START,
        "functional_initial_guess": "zero",
        "initial_guess": "zero_every_call",
        "production_warm_start_read": False,
        "production_warm_start_written": False,
        "transpose_action": "explicit_reverse_dataflow_conjugate_transpose",
        "pressure_transpose_protocol": (
            CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION
        ),
        "implicit_adjoint": CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION,
        "gradient": CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT,
    }


def _allocated_device(value: object) -> str:
    try:
        requested = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("functional execution device is invalid") from exc
    if requested.type != "cuda":
        return requested.type
    if not torch.cuda.is_available():
        raise RuntimeError("functional CUDA declaration requires CUDA")
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("functional CUDA device index is unavailable")
    return str(torch.device("cuda", index))


def _spectral_dtype(real_dtype: str) -> str:
    try:
        return {"float32": "complex64", "float64": "complex128"}[real_dtype]
    except KeyError as exc:  # pragma: no cover - public compiler rejects first
        raise ValueError("unsupported Channel functional precision") from exc


def _state_control_observation_specs(
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
    if simulation.numerics.spectral_storage.value != "full_complex":
        raise ValueError("P9.7.1 Channel declarations require full_complex storage")

    q_physical = FunctionalTensorSpec(
        name="q_physical",
        shape=(len(Q_COMPONENTS), 1, *shape),
        dtype=precision,
        device=device,
        batch_axis=1,
        layout="component_batch_xyz",
        meaning=(
            "physical five-component symmetric-traceless Channel Q state at "
            "the current timestep"
        ),
        component_names=Q_COMPONENTS,
    )
    q_spectral = FunctionalTensorSpec(
        name="q_spectral",
        shape=(len(Q_COMPONENTS), 1, *shape),
        dtype=_spectral_dtype(precision),
        device=device,
        batch_axis=1,
        layout="component_batch_periodic_x_neumann_y_neumann_z",
        meaning=(
            "full-complex native Channel Q spectrum retained as persistent "
            "projected-Euler state"
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
        meaning="cell-centered Channel activity coefficient alpha",
    )
    control = FunctionalControlFieldSpec(
        name=_ACTIVITY_NAME,
        tensor=activity_tensor,
        equation_term="div(beta * alpha * Q)",
        injection_order="form_beta_alpha_q_product_before_divergence",
        dealiasing_identity=(
            f"{simulation.numerics.dealias_rule.value}:"
            f"{simulation.numerics.projected_transform_execution.value}:"
            "channel_project_product_then_component_basis_divergence"
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
            meaning="physical compact Channel Q tensor at the input state",
            component_names=Q_COMPONENTS,
        ),
        convention="input physical compact Channel Q",
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
            meaning="physical no-slip Channel velocity at the input state",
            component_names=("ux", "uy", "uz"),
        ),
        convention="Channel Stokes velocity under current activity",
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
            meaning="zero-mean Channel pressure at the input state",
        ),
        convention="Channel pressure-PCG solution with zero-mean gauge",
        control_dependent=True,
        terminal_available_without_control=False,
    )
    return (
        FunctionalStateSpec((q_physical, q_spectral)),
        (control,),
        (q_observation, velocity_observation, pressure_observation),
    )


def channel_activity_functional_declaration(
    simulation: SimulationSpec,
) -> FunctionalRuntimeDeclaration:
    """Declare the qualified P9.7 Channel runtime without constructing it."""

    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    compiled = compile_public_simulation(simulation)
    if compiled.application != PUBLIC_CHANNEL_COMPLETE_APPLICATION:
        raise ValueError(
            "P9.7.1 requires the qualified complete-stress Channel application"
        )
    execution = simulation.execution.options
    if execution["pointwise_execution"] != "eager":
        raise ValueError("P9.7.1 Channel declarations require eager pointwise kernels")
    if simulation.numerics.precision.value != "float64":
        raise ValueError("P9.7.1 Channel declarations require float64")
    device = _allocated_device(execution["device"])
    state_spec, controls, observations = _state_control_observation_specs(
        simulation,
        device=device,
    )
    request = FunctionalRuntimeConstructionRequest(
        simulation=simulation,
        control_fields=controls,
        observations=observations,
        batch_size=1,
    )
    capabilities = channel_activity_functional_capabilities()

    identities = simulation.identity_metadata()
    execution_identity = dict(identities["execution"])
    execution_identity["functional_runtime_declaration"] = {
        "kind": CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
        "stage": "P9.7.4",
        "device": device,
        "executable": True,
        "pointwise_execution": "eager",
        "state_components": ["q_physical", "q_spectral"],
        "activity_product_before_divergence": True,
        "q_gradient_cache": "derived_rebuilt_not_state",
        "pressure_solver": channel_functional_pressure_identity(simulation),
        "boundary_spaces": {
            "Q": list(CHANNEL_Q_BOUNDARY_CONDITIONS),
            "velocity": list(CHANNEL_VELOCITY_BOUNDARY_CONDITIONS),
            "pressure": list(CHANNEL_PRESSURE_BOUNDARY_CONDITIONS),
        },
        "fallback_allowed": False,
        "fallback_used": False,
        "deterministic_replay": "bitwise",
        "durable_checkpoint_bridge_version": 1,
    }
    identity = FunctionalRuntimeIdentity(
        scientific=dict(identities["scientific"]),
        discretization=dict(identities["discretization"]),
        execution=execution_identity,
        state_layout=state_spec.to_metadata(),
    )
    return FunctionalRuntimeDeclaration(
        request=request,
        state_spec=state_spec,
        capabilities=capabilities,
        identity=identity,
        executable=True,
    )


def channel_activity_functional_request(
    simulation: SimulationSpec,
) -> FunctionalRuntimeConstructionRequest:
    """Return the canonical executable Channel construction request."""

    return channel_activity_functional_declaration(simulation).request


__all__ = [
    "CHANNEL_ACTIVITY_FUNCTIONAL_KIND",
    "CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT",
    "CHANNEL_FUNCTIONAL_PRESSURE_WARM_START",
    "channel_activity_functional_capabilities",
    "channel_activity_functional_declaration",
    "channel_activity_functional_request",
    "channel_functional_pressure_identity",
]
