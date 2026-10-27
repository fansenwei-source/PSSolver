"""Fail-closed P8.3 compiler for complete-stress rectangular Channels."""

from __future__ import annotations

from pssolver.configuration.channel_beris_edwards import (
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
    ChannelBerisEdwardsRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    channel_pressure_policy_metadata,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.public_complete_stress_validation import (
    validate_complete_stress_runtime_parameters,
    validate_existing_q_snapshot_shape,
)
from pssolver.configuration.public_option_validation import (
    ITERATIVE_PRESSURE_OPTION_KEYS,
    validate_iterative_pressure_options,
    validate_public_execution_device,
)
from pssolver.configuration.simulation import (
    InitialConditionSource,
    SimulationSpec,
)
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core.domain import GridPlacement
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.numerics import SpectralStorage
from pssolver.core._validation import require_exact_keys
from pssolver.core.pressure import (
    channel_pressure_minimum_relative_tolerance_for_precision,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS


_EXECUTION_KEYS = frozenset(
    {
        "device",
        "disable_q_gradient_reuse",
        "fallback_allowed",
        "molecular_field_linear_space",
        "pointwise_execution",
        "stress_divergence_sum_space",
        "tf32",
    }
)
_WORKFLOW_KEYS = frozenset(
    {
        "checkpoint_interval",
        "diagnostic_interval",
        "diagnostics",
        "output_dir",
        "restart_from",
        "save_hydrodynamics",
        "save_interval",
        "save_start_step",
    }
)
def _exact_keys(value, expected, description, *, code):
    from .public_simulation_runner import _reject

    return require_exact_keys(
        value,
        expected,
        description,
        reject=lambda message: _reject(message, code=code),
    )


def compile_channel_beris_edwards_public_simulation(
    source: SimulationSpec,
    product_type,
):
    """Compile the single P8.3 Channel application without allocation."""

    from .public_simulation_runner import (
        PublicCompilationRejectionCode as Code,
        _reject,
        _validate_invocation_options,
    )

    if (source.equation_system.variant, source.geometry.name) != (
        "complete_stress_beris_edwards",
        "rectangular_channel",
    ):
        raise AssertionError("complete-stress Channel compiler got wrong key")
    if source.geometry.domain.grid_placement is not GridPlacement.CELL_CENTERED:
        _reject(
            "complete-stress Channel currently requires a cell-centered grid",
            code=Code.GEOMETRY_CONTRACT,
        )
    if source.execution.backend != "torch_spectral":
        _reject(
            "complete-stress Channel requires torch_spectral",
            code=Code.EXECUTION_CONTRACT,
        )
    if source.execution.runtime_path != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
        _reject(
            "complete-stress Channel requires "
            "runtime_path='channel_complete_stress'",
            code=Code.EXECUTION_CONTRACT,
        )
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject(
            "P8.3 currently requires projected Euler",
            code=Code.TIME_INTEGRATION_CONTRACT,
        )
    if source.initial_condition.source is not InitialConditionSource.SNAPSHOT:
        _reject(
            "P8.3 requires an external Q snapshot",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    if source.initial_condition.family != "external_snapshot":
        _reject(
            "P8.3 snapshot family must be external_snapshot",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    snapshot = dict(source.initial_condition.parameters)
    if set(snapshot) != {"directory", "step", "mode"}:
        _reject(
            "P8.3 snapshot requires directory, step, and mode",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    if snapshot["mode"] != "branch":
        _reject(
            "P8.3 accepts snapshot mode='branch'; restart uses checkpoints",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    try:
        validate_existing_q_snapshot_shape(
            snapshot,
            tuple(source.geometry.domain.shape),
            component_count=len(Q_COMPONENTS),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _reject(str(exc), code=Code.INITIAL_CONDITION_CONTRACT)

    execution = _exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "complete-stress Channel execution options",
        code=Code.EXECUTION_CONTRACT,
    )
    if execution["fallback_allowed"] is not False:
        _reject(
            "complete-stress Channel fallback must remain disabled",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["tf32"] != "off":
        _reject(
            "P8.3 qualifies strict TF32-off execution only",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["molecular_field_linear_space"] not in {
        "physical",
        "spectral",
    }:
        _reject(
            "unsupported Channel molecular-field linear space",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["stress_divergence_sum_space"] != "physical":
        _reject(
            "two-bounded-axis complete stress requires physical-space "
            "component-basis divergence summation",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["pointwise_execution"] not in {"eager", "compile"}:
        _reject(
            "unsupported Channel pointwise execution mode",
            code=Code.EXECUTION_CONTRACT,
        )
    try:
        device_normalization = validate_public_execution_device(
            execution["device"]
        )
    except ValueError as exc:
        _reject(str(exc), code=Code.EXECUTION_CONTRACT)
    if not isinstance(execution["disable_q_gradient_reuse"], bool):
        _reject(
            "Channel Q-gradient reuse flag must be a bool",
            code=Code.EXECUTION_CONTRACT,
        )
    _exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "complete-stress Channel workflow options",
        code=Code.OUTPUT_CONTRACT,
    )
    _validate_invocation_options(
        source.invocation.options,
        description="Channel invocation options",
    )
    discretization = _exact_keys(
        source.discretization_parameters,
        frozenset({"pressure_solver"}),
        "complete-stress Channel discretization",
        code=Code.DISCRETIZATION_CONTRACT,
    )
    pressure = _exact_keys(
        discretization["pressure_solver"],
        ITERATIVE_PRESSURE_OPTION_KEYS,
        "Channel pressure solver",
        code=Code.DISCRETIZATION_CONTRACT,
    )
    try:
        pressure = validate_iterative_pressure_options(
            pressure,
            description="Channel pressure solver",
            minimum_relative_tolerance=(
                channel_pressure_minimum_relative_tolerance_for_precision(
                    source.numerics.precision.value
                )
            ),
        )
    except ValueError as exc:
        _reject(str(exc), code=Code.DISCRETIZATION_CONTRACT)
    if pressure["warm_start"] is not True:
        _reject(
            "P8.3 requires pressure-PCG warm start",
            code=Code.DISCRETIZATION_CONTRACT,
        )
    if source.time_integration.refresh != {"mode": "disabled"}:
        _reject(
            "P8.3 requires disabled spectral refresh",
            code=Code.TIME_INTEGRATION_CONTRACT,
        )
    if source.numerics.spectral_storage is not SpectralStorage.FULL_COMPLEX:
        _reject(
            "P8.3 initially qualifies full_complex spectral storage",
            code=Code.NUMERICS_CONTRACT,
        )

    validate_complete_stress_runtime_parameters(
        source.equation_system.parameters
    )
    stokes = source.equation_system.parameters["stokes"]["parameters"]
    if stokes["pressure_gauge"] != "zero_mean":
        _reject(
            "Channel pressure requires the zero-mean gauge",
            code=Code.MODEL_CONTRACT,
        )

    run_spec = ChannelBerisEdwardsRunSpec(source)
    return product_type(
        application="channel_complete_stress_beris_edwards",
        source_simulation_sha256=source.canonical_sha256(),
        application_simulation=source,
        lowering_plan=lower_simulation_spec(source),
        construction_plan=plan_package_runtime_construction(source),
        run_spec=run_spec,
        normalization={
            "schema_version": 1,
            "physical_values_changed": False,
            "pressure_gauge": "zero_mean",
            "declared_tangential_zero_mode_policy": stokes[
                "tangential_zero_mode_policy"
            ],
            "effective_tangential_zero_mode_policy": "not_applicable",
            "velocity_nullspace": (
                "absent_due_to_two_bounded_no_slip_axes"
            ),
            "requested_molecular_field_linear_space": execution[
                "molecular_field_linear_space"
            ],
            "effective_molecular_field_linear_space": execution[
                "molecular_field_linear_space"
            ],
            "molecular_field_space_fallback_used": False,
            "runtime_fallback_allowed": False,
            **device_normalization,
            "pressure_solver_policy": channel_pressure_policy_metadata(
                relative_tolerance=pressure["relative_tolerance"],
                max_iterations=pressure["max_iterations"],
                fixed_iterations=pressure["fixed_iterations"],
                initial_guess_policy=(
                    "persistent_production_warm_start"
                ),
            ),
        },
    )


__all__ = ["compile_channel_beris_edwards_public_simulation"]
