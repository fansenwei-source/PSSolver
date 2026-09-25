"""Qualified public compiler adapter for the existing Channel application."""

from __future__ import annotations

from collections.abc import Mapping

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
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PublicSimulationCompilation,
    PublicSimulationCompilationError,
)
from pssolver.configuration.simulation import (
    InitialConditionSource,
    SimulationSpec,
)
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core.integrators import IntegratorScheme


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
_WORKFLOW_KEYS = frozenset(
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
_PRESSURE_KEYS = frozenset(
    {
        "algorithm",
        "fixed_iterations",
        "max_iterations",
        "relative_tolerance",
        "warm_start",
    }
)


def _reject(message: str) -> None:
    raise PublicSimulationCompilationError(message)


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    description: str,
) -> dict[str, object]:
    observed = set(value)
    if observed != expected:
        _reject(
            f"{description} must declare every qualified option explicitly; "
            f"missing={tuple(sorted(expected - observed))!r}, "
            f"extra={tuple(sorted(observed - expected))!r}"
        )
    return dict(value)


def _validate_translation(
    source: SimulationSpec,
    application: SimulationSpec,
) -> None:
    comparisons = (
        ("equation system", source.equation_system, application.equation_system),
        ("geometry", source.geometry, application.geometry),
        (
            "boundary laws",
            source.boundaries.to_metadata(),
            application.boundaries.to_metadata(),
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
            source.workflow.to_metadata(),
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
            _reject(f"public {description} changed during Channel translation")


def compile_channel_public_simulation(
    source: SimulationSpec,
) -> PublicSimulationCompilation:
    """Translate the qualified Channel declaration to its existing RunSpec.

    This adapter is intentionally exact: it normalizes no physical value and
    supplies no hidden default.  The source must fully declare the already
    qualified Channel application contract.
    """

    if not isinstance(source, SimulationSpec):
        raise TypeError("source must be a SimulationSpec")
    if (
        source.equation_system.variant,
        source.geometry.name,
    ) != ("legacy_active_force_active_nematics", "rectangular_channel"):
        raise AssertionError("Channel compiler received the wrong registry key")
    if source.execution.backend != "torch_spectral":
        _reject("only the torch_spectral Channel backend is connected")
    if source.execution.runtime_path not in {
        "legacy_channel",
        "compiled_channel_v2",
    }:
        _reject(
            "public Channel execution supports legacy_channel or "
            "compiled_channel_v2"
        )
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject("the public Channel compiler currently requires projected Euler")
    if source.time_integration.refresh:
        _reject("the qualified Channel application does not use spectral refresh")
    if source.invocation.options:
        _reject("the qualified Channel application accepts no invocation options")

    initial = _exact_keys(
        source.initial_condition.parameters,
        _INITIAL_KEYS,
        "Channel initial-condition parameters",
    )
    generated = _exact_keys(
        initial["generated"],
        _GENERATED_KEYS,
        "Channel generated initial-condition parameters",
    )
    snapshot = _exact_keys(
        initial["snapshot"],
        _SNAPSHOT_KEYS,
        "Channel snapshot initial-condition parameters",
    )
    mode = initial["mode"]
    if mode not in {"generated", "snapshot"}:
        _reject("Channel initial-condition mode must be generated or snapshot")
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
        _reject("Channel initial-condition source disagrees with its mode")
    if source.initial_condition.family != expected_family:
        _reject("Channel initial-condition family disagrees with its mode")
    if generated["name"] != "aligned_x_smooth_noise":
        _reject("the qualified Channel generator is aligned_x_smooth_noise")

    execution = _exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "Channel execution options",
    )
    if execution["fallback_allowed"] is not False:
        _reject("runtime fallback must remain disabled")
    if execution["dtype"] != source.numerics.precision.value:
        _reject("Channel execution dtype disagrees with numerical precision")
    if execution["dtype"] != "float32":
        _reject("the qualified Channel production application requires float32")
    if execution["batch_size"] != 1:
        _reject("the qualified Channel production application requires batch_size=1")
    workflow = _exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "Channel workflow options",
    )
    discretization = _exact_keys(
        source.discretization_parameters,
        frozenset({"pressure_solver"}),
        "Channel discretization",
    )
    pressure = _exact_keys(
        discretization["pressure_solver"],
        _PRESSURE_KEYS,
        "Channel pressure solver",
    )
    if pressure["algorithm"] != "preconditioned_conjugate_gradient":
        _reject("the qualified Channel pressure solver is PCG")
    if pressure["warm_start"] is not True:
        _reject("the qualified Channel pressure solver uses a warm start")

    parameters = source.equation_system.parameters
    if parameters["flow_alignment"] != 1.0:
        _reject(
            "the qualified Channel production application requires "
            "flow_alignment=1"
        )
    stokes = dict(parameters["stokes"]["parameters"])
    if stokes["pressure_gauge"] != "zero_mean":
        _reject("the qualified Channel pressure gauge is zero_mean")
    if stokes["tangential_zero_mode_policy"] != "not_applicable":
        _reject("Channel tangential zero-mode policy must be not_applicable")

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
    _validate_translation(source, application_simulation)
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
        },
    )


__all__ = ["compile_channel_public_simulation"]
