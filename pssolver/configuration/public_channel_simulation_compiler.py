"""Qualified public compiler adapter for the existing Channel application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    CHANNEL_BOUNDARIES,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.public_option_validation import (
    ITERATIVE_PRESSURE_OPTION_KEYS,
    validate_iterative_pressure_options,
    validate_public_execution_device,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PublicCompilationRejectionCode,
    PublicSimulationCompilation,
    PublicSimulationCompilationError,
)
from pssolver.configuration.simulation import (
    InitialConditionSource,
    SimulationSpec,
    WorkflowSpec,
)
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.pressure import (
    channel_pressure_minimum_relative_tolerance_for_precision,
)


_INITIAL_KEYS = frozenset({"generated", "mode", "snapshot"})
_GENERATED_KEYS = frozenset(
    {
        "initial_s",
        "name",
        "noise_phi",
        "noise_theta",
        "seed",
        "smoothing_sigma",
    }
)
_SNAPSHOT_KEYS = frozenset({"directory", "mode", "step"})
_EXECUTION_KEYS = frozenset(
    {"batch_size", "device", "dtype", "fallback_allowed"}
)
_LEGACY_WORKFLOW_KEYS = frozenset(
    {
        "checkpoint_interval",
        "diagnostic_interval",
        "diagnostics_enabled",
        "generated_output_directory",
        "restart_from",
        "save_interval",
        "snapshot_output_directory",
    }
)
_COMMON_OUTPUT_KEYS = frozenset(
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
def _reject(
    message: str,
    *,
    code: PublicCompilationRejectionCode,
) -> None:
    raise PublicSimulationCompilationError(message, code=code)


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    description: str,
    *,
    code: PublicCompilationRejectionCode,
) -> dict[str, object]:
    observed = set(value)
    if observed != expected:
        _reject(
            f"{description} must declare every qualified option explicitly; "
            f"missing={tuple(sorted(expected - observed))!r}, "
            f"extra={tuple(sorted(observed - expected))!r}",
            code=code,
        )
    return dict(value)


def _validate_translation(
    source: SimulationSpec,
    application: SimulationSpec,
    *,
    expected_workflow: WorkflowSpec | None = None,
) -> None:
    source_boundaries = source.boundaries.to_metadata()
    application_boundaries = application.boundaries.to_metadata()
    comparisons = (
        ("equation system", source.equation_system, application.equation_system),
        ("geometry", source.geometry, application.geometry),
        (
            "boundary laws",
            {
                "ndim": source_boundaries["ndim"],
                "components": source_boundaries["components"],
            },
            {
                "ndim": application_boundaries["ndim"],
                "components": application_boundaries["components"],
            },
        ),
        (
            "numerical policy",
            source.numerics.to_metadata(),
            application.numerics.to_metadata(),
        ),
        (
            "time integration",
            source.time_integration.to_metadata(),
            application.time_integration.to_metadata(),
        ),
        (
            "discretization",
            dict(source.discretization_parameters),
            dict(application.discretization_parameters),
        ),
        (
            "initial condition",
            source.initial_condition.to_metadata(),
            application.initial_condition.to_metadata(),
        ),
        (
            "execution",
            source.execution.to_metadata(),
            application.execution.to_metadata(),
        ),
        (
            "workflow",
            (
                source.workflow
                if expected_workflow is None
                else expected_workflow
            ).to_metadata(),
            application.workflow.to_metadata(),
        ),
        (
            "invocation",
            source.invocation.to_metadata(),
            application.invocation.to_metadata(),
        ),
    )
    for description, requested, effective in comparisons:
        if requested != effective:
            code = {
                "equation system": PublicCompilationRejectionCode.MODEL_CONTRACT,
                "geometry": PublicCompilationRejectionCode.GEOMETRY_CONTRACT,
                "boundary laws": PublicCompilationRejectionCode.BOUNDARY_CONTRACT,
                "numerical policy": PublicCompilationRejectionCode.NUMERICS_CONTRACT,
                "time integration": (
                    PublicCompilationRejectionCode.TIME_INTEGRATION_CONTRACT
                ),
                "discretization": (
                    PublicCompilationRejectionCode.DISCRETIZATION_CONTRACT
                ),
                "initial condition": (
                    PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT
                ),
                "execution": PublicCompilationRejectionCode.EXECUTION_CONTRACT,
                "workflow": PublicCompilationRejectionCode.OUTPUT_CONTRACT,
                "invocation": PublicCompilationRejectionCode.INVOCATION_CONTRACT,
            }[description]
            _reject(
                f"public {description} changed during Channel translation",
                code=code,
            )


def _adapt_workflow(
    source: WorkflowSpec,
) -> tuple[dict[str, object], WorkflowSpec, bool]:
    """Adapt the geometry-neutral Output contract to legacy Channel keys."""

    observed = set(source.options)
    if observed == _LEGACY_WORKFLOW_KEYS:
        return dict(source.options), source, False
    if observed != _COMMON_OUTPUT_KEYS:
        _reject(
            "Channel workflow options must match either the public Output "
            "contract or the legacy Channel workflow contract; "
            f"observed={tuple(sorted(observed))!r}",
            code=PublicCompilationRejectionCode.OUTPUT_CONTRACT,
        )
    common = dict(source.options)
    if common["save_start_step"] != 0:
        _reject(
            "legacy Channel public Output requires save_start_step=0",
            code=PublicCompilationRejectionCode.OUTPUT_CONTRACT,
        )
    if common["save_hydrodynamics"] is not True:
        _reject(
            "legacy Channel always saves hydrodynamics; "
            "save_hydrodynamics must be true",
            code=PublicCompilationRejectionCode.OUTPUT_CONTRACT,
        )
    output_directory = common["output_dir"]
    adapted = {
        "checkpoint_interval": common["checkpoint_interval"],
        "diagnostic_interval": common["diagnostic_interval"],
        "diagnostics_enabled": common["diagnostics"],
        "generated_output_directory": output_directory,
        "restart_from": common["restart_from"],
        "save_interval": common["save_interval"],
        # Snapshot initialization is rejected before this adaptation.  Keep a
        # deterministic value for the legacy schema without exposing a second
        # directory in the geometry-neutral public API.
        "snapshot_output_directory": output_directory,
    }
    return adapted, WorkflowSpec(source.steps, adapted), True


def compile_channel_public_simulation(
    source: SimulationSpec,
) -> PublicSimulationCompilation:
    """Translate the qualified Channel declaration to its existing RunSpec.

    This adapter normalizes no physical value and supplies no hidden default.
    It does translate the geometry-neutral public ``Output`` schema into the
    legacy Channel workflow's historical directory-key schema.
    """

    if not isinstance(source, SimulationSpec):
        raise TypeError("source must be a SimulationSpec")
    if (
        source.equation_system.variant,
        source.geometry.name,
    ) != ("legacy_active_force_active_nematics", "rectangular_channel"):
        raise AssertionError("Channel compiler received the wrong registry key")
    if source.execution.backend != "torch_spectral":
        _reject(
            "only the torch_spectral Channel backend is connected",
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    if source.execution.runtime_path not in {
        "legacy_channel",
        "compiled_channel_v2",
    }:
        _reject(
            "public Channel execution supports legacy_channel or "
            "compiled_channel_v2",
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject(
            "the public Channel compiler currently requires projected Euler",
            code=PublicCompilationRejectionCode.TIME_INTEGRATION_CONTRACT,
        )
    if source.time_integration.refresh != {"mode": "disabled"}:
        _reject(
            "the qualified Channel application requires disabled spectral refresh",
            code=PublicCompilationRejectionCode.TIME_INTEGRATION_CONTRACT,
        )
    invocation = dict(source.invocation.options)
    if set(invocation) - {"dry_run"}:
        _reject(
            "the qualified Channel application accepts only the dry_run "
            "invocation option",
            code=PublicCompilationRejectionCode.INVOCATION_CONTRACT,
        )
    if "dry_run" in invocation and not isinstance(invocation["dry_run"], bool):
        _reject(
            "Channel dry_run must be a bool",
            code=PublicCompilationRejectionCode.INVOCATION_CONTRACT,
        )

    initial = _exact_keys(
        source.initial_condition.parameters,
        _INITIAL_KEYS,
        "Channel initial-condition parameters",
        code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
    )
    generated = _exact_keys(
        initial["generated"],
        _GENERATED_KEYS,
        "Channel generated initial-condition parameters",
        code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
    )
    snapshot = _exact_keys(
        initial["snapshot"],
        _SNAPSHOT_KEYS,
        "Channel snapshot initial-condition parameters",
        code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
    )
    mode = initial["mode"]
    if mode not in {"generated", "snapshot"}:
        _reject(
            "Channel initial-condition mode must be generated or snapshot",
            code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
        )
    if mode == "snapshot":
        _reject(
            "the public legacy Channel application does not implement "
            "snapshot initial conditions; use a workflow checkpoint to resume",
            code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
        )
    expected_source = (
        InitialConditionSource.GENERATED
        if mode == "generated"
        else InitialConditionSource.SNAPSHOT
    )
    expected_family = (
        "aligned_x_smooth_noise"
        if mode == "generated"
        else "external_snapshot"
    )
    if source.initial_condition.source is not expected_source:
        _reject(
            "Channel initial-condition source disagrees with its mode",
            code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
        )
    if source.initial_condition.family != expected_family:
        _reject(
            "Channel initial-condition family disagrees with its mode",
            code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
        )
    if generated["name"] != "aligned_x_smooth_noise":
        _reject(
            "the qualified Channel generator is aligned_x_smooth_noise",
            code=PublicCompilationRejectionCode.INITIAL_CONDITION_CONTRACT,
        )

    execution = _exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "Channel execution options",
        code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
    )
    if execution["fallback_allowed"] is not False:
        _reject(
            "runtime fallback must remain disabled",
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    if execution["dtype"] != source.numerics.precision.value:
        _reject(
            "Channel execution dtype disagrees with numerical precision",
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    if execution["dtype"] != "float32":
        _reject(
            "the qualified Channel production application requires float32",
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    if execution["batch_size"] != 1:
        _reject(
            "the qualified Channel production application requires batch_size=1",
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    try:
        device_normalization = validate_public_execution_device(
            execution["device"]
        )
    except ValueError as exc:
        _reject(
            str(exc),
            code=PublicCompilationRejectionCode.EXECUTION_CONTRACT,
        )
    workflow, effective_workflow, workflow_adapted = _adapt_workflow(
        source.workflow
    )
    discretization = _exact_keys(
        source.discretization_parameters,
        frozenset({"pressure_solver"}),
        "Channel discretization",
        code=PublicCompilationRejectionCode.DISCRETIZATION_CONTRACT,
    )
    pressure = _exact_keys(
        discretization["pressure_solver"],
        ITERATIVE_PRESSURE_OPTION_KEYS,
        "Channel pressure solver",
        code=PublicCompilationRejectionCode.DISCRETIZATION_CONTRACT,
    )
    try:
        pressure = validate_iterative_pressure_options(
            pressure,
            description="Channel pressure solver",
            minimum_relative_tolerance=(
                channel_pressure_minimum_relative_tolerance_for_precision(
                    execution["dtype"]
                )
            ),
        )
    except ValueError as exc:
        _reject(
            str(exc),
            code=PublicCompilationRejectionCode.DISCRETIZATION_CONTRACT,
        )
    if pressure["warm_start"] is not True:
        _reject(
            "the qualified Channel pressure solver uses a warm start",
            code=PublicCompilationRejectionCode.DISCRETIZATION_CONTRACT,
        )

    parameters = source.equation_system.parameters
    if parameters["flow_alignment"] != 1.0:
        _reject(
            "the qualified Channel production application requires "
            "flow_alignment=1",
            code=PublicCompilationRejectionCode.MODEL_CONTRACT,
        )
    stokes = dict(parameters["stokes"]["parameters"])
    if stokes["pressure_gauge"] != "zero_mean":
        _reject(
            "the qualified Channel pressure gauge is zero_mean",
            code=PublicCompilationRejectionCode.MODEL_CONTRACT,
        )
    if stokes["tangential_zero_mode_policy"] != "not_applicable":
        _reject(
            "Channel tangential zero-mode policy must be not_applicable",
            code=PublicCompilationRejectionCode.MODEL_CONTRACT,
        )

    domain = source.geometry.domain
    numerics = source.numerics
    run_spec = ChannelActiveNematicRunSpec(
        shape=domain.shape,
        lengths=domain.lengths,
        dt=source.time_integration.integrator.dt,
        steps=source.workflow.steps,
        save_interval=workflow["save_interval"],
        diagnostics_enabled=workflow["diagnostics_enabled"],
        diagnostic_interval=workflow["diagnostic_interval"],
        batch_size=execution["batch_size"],
        checkpoint_interval=workflow["checkpoint_interval"],
        restart_from=workflow["restart_from"],
        seed=generated["seed"],
        rho=parameters["rho"],
        elastic_constant=parameters["elastic_constant"],
        activity=parameters["activity"],
        beta=parameters["beta"],
        friction=stokes["friction"],
        viscosity=stokes["viscosity"],
        flow_alignment=parameters["flow_alignment"],
        initial_s=generated["initial_s"],
        noise_theta=generated["noise_theta"],
        noise_phi=generated["noise_phi"],
        smoothing_sigma=tuple(generated["smoothing_sigma"]),
        initialization_mode=mode,
        snapshot_mode=snapshot["mode"],
        snapshot_directory=snapshot["directory"],
        snapshot_step=snapshot["step"],
        generated_output_directory=workflow["generated_output_directory"],
        snapshot_output_directory=workflow["snapshot_output_directory"],
        pressure_relative_tolerance=pressure["relative_tolerance"],
        pressure_max_iterations=pressure["max_iterations"],
        pressure_fixed_iterations=pressure["fixed_iterations"],
        device=execution["device"],
        dtype=execution["dtype"],
        dealias_rule=numerics.dealias_rule.value,
        projected_transform_execution=(
            numerics.projected_transform_execution.value
        ),
        transform_execution_order=numerics.transform_execution_order.value,
        spectral_storage=numerics.spectral_storage.value,
        runtime_path=source.execution.runtime_path,
        boundaries=CHANNEL_BOUNDARIES,
    )
    application_simulation = compose_channel_active_nematics_simulation(
        run_spec.components
    )
    application_simulation = replace(
        application_simulation,
        invocation=source.invocation,
    )
    _validate_translation(
        source,
        application_simulation,
        expected_workflow=effective_workflow,
    )
    lowering_plan = lower_simulation_spec(application_simulation)
    construction_plan = plan_package_runtime_construction(
        application_simulation
    )
    return PublicSimulationCompilation(
        application=PUBLIC_CHANNEL_APPLICATION,
        source_simulation_sha256=source.canonical_sha256(),
        application_simulation=application_simulation,
        lowering_plan=lowering_plan,
        construction_plan=construction_plan,
        run_spec=run_spec,
        normalization={
            "schema_version": 1,
            "adapter": "exact_channel_run_spec",
            "coefficient_parameterization": "rho",
            "physical_values_changed": False,
            "runtime_fallback_allowed": False,
            **device_normalization,
            "workflow_adapter": (
                "public_output_to_legacy_channel"
                if workflow_adapted
                else "legacy_channel_exact"
            ),
        },
    )


__all__ = ["compile_channel_public_simulation"]
