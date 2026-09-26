"""Fail-closed P8.3 compiler for complete-stress rectangular Channels."""

from __future__ import annotations

from collections.abc import Mapping

from pssolver.configuration.channel_beris_edwards import (
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
    ChannelBerisEdwardsRunSpec,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.simulation import (
    InitialConditionSource,
    SimulationSpec,
)
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.numerics import SpectralStorage


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
_INVOCATION_KEYS = frozenset({"dry_run", "validation_config_sha256"})
_PRESSURE_KEYS = frozenset(
    {
        "algorithm",
        "fixed_iterations",
        "max_iterations",
        "relative_tolerance",
        "warm_start",
    }
)


def _exact_keys(value: Mapping[str, object], expected, description):
    from .public_simulation_runner import _reject

    observed = set(value)
    if observed != expected:
        _reject(
            f"{description} must declare every qualified option explicitly; "
            f"missing={tuple(sorted(expected - observed))!r}, "
            f"extra={tuple(sorted(observed - expected))!r}"
        )
    return dict(value)


def compile_channel_beris_edwards_public_simulation(
    source: SimulationSpec,
    product_type,
):
    """Compile the single P8.3 Channel application without allocation."""

    from .public_simulation_runner import _reject

    if (source.equation_system.variant, source.geometry.name) != (
        "complete_stress_beris_edwards",
        "rectangular_channel",
    ):
        raise AssertionError("complete-stress Channel compiler got wrong key")
    if source.execution.backend != "torch_spectral":
        _reject("complete-stress Channel requires torch_spectral")
    if source.execution.runtime_path != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
        _reject(
            "complete-stress Channel requires "
            "runtime_path='channel_complete_stress'"
        )
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject("P8.3 currently requires projected Euler")
    if source.initial_condition.source is not InitialConditionSource.SNAPSHOT:
        _reject("P8.3 requires an external Q snapshot")
    if source.initial_condition.family != "external_snapshot":
        _reject("P8.3 snapshot family must be external_snapshot")
    snapshot = dict(source.initial_condition.parameters)
    if set(snapshot) != {"directory", "step", "mode"}:
        _reject("P8.3 snapshot requires directory, step, and mode")
    if snapshot["mode"] != "branch":
        _reject("P8.3 accepts snapshot mode='branch'; restart uses checkpoints")

    execution = _exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "complete-stress Channel execution options",
    )
    if execution["fallback_allowed"] is not False:
        _reject("complete-stress Channel fallback must remain disabled")
    if execution["tf32"] != "off":
        _reject("P8.3 qualifies strict TF32-off execution only")
    if execution["molecular_field_linear_space"] not in {
        "physical",
        "spectral",
    }:
        _reject("unsupported Channel molecular-field linear space")
    if execution["stress_divergence_sum_space"] != "physical":
        _reject(
            "two-bounded-axis complete stress requires physical-space "
            "component-basis divergence summation"
        )
    if execution["pointwise_execution"] not in {"eager", "compile"}:
        _reject("unsupported Channel pointwise execution mode")
    _exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "complete-stress Channel workflow options",
    )
    invocation = dict(source.invocation.options)
    if not set(invocation) <= _INVOCATION_KEYS:
        _reject(
            "Channel invocation options may contain only dry_run and "
            "validation_config_sha256"
        )
    discretization = _exact_keys(
        source.discretization_parameters,
        frozenset({"pressure_solver"}),
        "complete-stress Channel discretization",
    )
    pressure = _exact_keys(
        discretization["pressure_solver"],
        _PRESSURE_KEYS,
        "Channel pressure solver",
    )
    if pressure["algorithm"] != "preconditioned_conjugate_gradient":
        _reject("P8.3 requires the Channel pressure PCG")
    if pressure["warm_start"] is not True:
        _reject("P8.3 requires pressure-PCG warm start")
    if source.time_integration.refresh != {"mode": "disabled"}:
        _reject("P8.3 requires disabled spectral refresh")
    if source.numerics.spectral_storage is not SpectralStorage.FULL_COMPLEX:
        _reject("P8.3 initially qualifies full_complex spectral storage")

    stokes = source.equation_system.parameters["stokes"]["parameters"]
    if stokes["pressure_gauge"] != "zero_mean":
        _reject("Channel pressure requires the zero-mean gauge")

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
            "runtime_fallback_allowed": False,
        },
    )


__all__ = ["compile_channel_beris_edwards_public_simulation"]
