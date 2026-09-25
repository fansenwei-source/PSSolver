"""Fail-closed P8.2 compiler for complete-stress periodic boxes."""

from __future__ import annotations

from collections.abc import Mapping

from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.periodic_beris_edwards import (
    PERIODIC_RUNTIME_PATH,
    PeriodicBerisEdwardsRunSpec,
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


def compile_periodic_public_simulation(source: SimulationSpec, product_type):
    """Compile the single P8.2 periodic application without allocation."""

    from .public_simulation_runner import _reject

    if (source.equation_system.variant, source.geometry.name) != (
        "complete_stress_beris_edwards",
        "periodic_box",
    ):
        raise AssertionError("periodic compiler received the wrong registry key")
    if source.execution.backend != "torch_spectral":
        _reject("periodic execution requires the torch_spectral backend")
    if source.execution.runtime_path != PERIODIC_RUNTIME_PATH:
        _reject("periodic execution requires runtime_path='periodic_spectral'")
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject("periodic execution currently requires projected Euler")
    if source.initial_condition.source is not InitialConditionSource.SNAPSHOT:
        _reject("P8.2 periodic execution requires an external Q snapshot")
    if source.initial_condition.family != "external_snapshot":
        _reject("periodic snapshot family must be external_snapshot")
    snapshot = dict(source.initial_condition.parameters)
    if set(snapshot) != {"directory", "step", "mode"}:
        _reject("periodic snapshot requires directory, step, and mode")
    if snapshot["mode"] != "branch":
        _reject("P8.2 accepts snapshot mode='branch'; restart uses checkpoints")
    execution = _exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "periodic execution options",
    )
    if execution["fallback_allowed"] is not False:
        _reject("periodic runtime fallback must remain disabled")
    if execution["tf32"] != "off":
        _reject("P8.2 qualifies strict TF32-off execution only")
    if execution["molecular_field_linear_space"] not in {
        "physical",
        "spectral",
    }:
        _reject("unsupported periodic molecular-field linear space")
    if execution["stress_divergence_sum_space"] not in {
        "physical",
        "spectral",
    }:
        _reject("unsupported periodic stress-divergence sum space")
    if execution["pointwise_execution"] not in {"eager", "compile"}:
        _reject("unsupported periodic pointwise execution mode")
    _exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "periodic workflow options",
    )
    invocation = dict(source.invocation.options)
    if not set(invocation) <= _INVOCATION_KEYS:
        _reject(
            "periodic invocation options may contain only dry_run and "
            "validation_config_sha256"
        )
    if source.discretization_parameters:
        _reject("periodic execution accepts no discretization overrides")
    if source.time_integration.refresh != {"mode": "disabled"}:
        _reject("P8.2 periodic execution requires disabled spectral refresh")
    if (
        source.numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
        and source.numerics.hermitian_axis not in source.geometry.periodic_axes
    ):
        _reject("periodic Hermitian packing must use a periodic axis")
    stokes = source.equation_system.parameters["stokes"]["parameters"]
    if stokes["pressure_gauge"] != "zero_mean":
        _reject("periodic pressure requires the zero-mean gauge")
    if stokes["tangential_zero_mode_policy"] != "zero_mean":
        _reject("P8.2 qualifies zero-mean uniform velocity only")
    if stokes["friction"] != 0.0:
        _reject("zero-mean periodic velocity requires zero friction")

    run_spec = PeriodicBerisEdwardsRunSpec(source)
    return product_type(
        application="periodic_complete_stress_beris_edwards",
        source_simulation_sha256=source.canonical_sha256(),
        application_simulation=source,
        lowering_plan=lower_simulation_spec(source),
        construction_plan=plan_package_runtime_construction(source),
        run_spec=run_spec,
        normalization={
            "schema_version": 1,
            "physical_values_changed": False,
            "pressure_gauge": "zero_mean",
            "uniform_velocity_mode_action": (
                "remove_all_uniform_velocity_and_force"
            ),
            "runtime_fallback_allowed": False,
        },
    )


__all__ = ["compile_periodic_public_simulation"]
