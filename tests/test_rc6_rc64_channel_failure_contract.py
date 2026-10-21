"""RC6-4 public Channel failure hierarchy and failed-run metadata."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pssolver import (
    ChannelPressureConvergenceError,
    SimulationExecutionError,
    SimulationStateError,
)
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.functional.errors import (
    FunctionalConvergenceError,
    translate_functional_exception,
)
from pssolver.workflows.channel_active_nematics import (
    ChannelActiveNematicsWorkflow,
)
from pssolver.workflows.plane_beris_edwards import (
    PlaneBerisEdwardsWorkflow,
)


class _FailingPlaneAdapter:
    runtime_path = PlaneRuntimePath.LEGACY_PRODUCTION

    def __init__(self, error: SimulationExecutionError) -> None:
        self._error = error
        self.completed_steps = 0
        self.solver = SimpleNamespace(
            integrator=SimpleNamespace(refresh_count=0)
        )
        self.projector = object()
        self.fields = {}

    def advance(self, steps, *, pre_update_callback=None):
        raise self._error

    def synchronize_for_observation(self):
        return None

    def projected_normal_force(self):
        return None

    def flow_diagnostics(self):
        return {}

    def backend_restart_metadata(self):
        return {}

    def restore_progress(self, **_kwargs):
        return None

    def to_metadata(self):
        return {}


class _FailingChannelAdapter:
    runtime_path = ChannelRuntimePath.LEGACY_CHANNEL

    def __init__(self, error: SimulationExecutionError) -> None:
        self._error = error
        self.completed_steps = 0
        self.solver = object()
        self.fields = {}
        self.output_views = object()

    def advance(self, steps, *, pre_update_callback=None):
        raise self._error

    def synchronize_for_observation(self):
        return None

    def capture_pressure_guess(self):
        return None

    def restore_pressure_guess(self, pressure_guess):
        return None

    def backend_restart_metadata(self):
        return {}

    def restore_progress(self, **_kwargs):
        return None

    def to_metadata(self):
        return {}


def _plane_spec(output: Path):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=output,
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=8,
        ny=8,
        nz=8,
        steps=1,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        checkpoint_interval=None,
        diagnostics=True,
        save_hydrodynamics=True,
        spectral_refresh_time=None,
        disable_spectral_refresh=True,
    )


def test_pressure_convergence_error_uses_public_execution_hierarchy():
    error = ChannelPressureConvergenceError(
        "pressure input contains NaN or Inf",
        details={"acceptable": False, "termination_reason": "nonfinite_input"},
    )

    assert isinstance(error, SimulationExecutionError)
    assert error.application_family == "channel_pressure"
    assert error.stage == "pressure_solve"
    assert error.step == 0
    assert error.to_metadata() == {
        "code": "channel_pressure_convergence_error",
        "message": "pressure input contains NaN or Inf",
        "application_family": "channel_pressure",
        "stage": "pressure_solve",
        "step": 0,
        "details": {
            "acceptable": False,
            "termination_reason": "nonfinite_input",
        },
    }


def test_functional_pressure_translation_retains_both_public_hierarchies():
    source = ChannelPressureConvergenceError(
        "pressure solve failed",
        details={"acceptable": False},
    )

    translated = translate_functional_exception(source, operation="step")

    assert isinstance(translated, FunctionalConvergenceError)
    assert isinstance(translated, ChannelPressureConvergenceError)
    assert isinstance(translated, SimulationExecutionError)
    assert translated.details == {"acceptable": False}


def test_functional_convergence_error_has_complete_finite_execution_metadata():
    error = FunctionalConvergenceError(
        "pressure solve failed",
        operation="step",
        details={
            "residual": float("nan"),
            "bounds": [float("-inf"), float("inf")],
        },
    )

    assert isinstance(error, SimulationExecutionError)
    assert error.application_family == "channel_pressure"
    assert error.stage == "pressure_solve"
    assert error.step == 0
    assert error.details == {
        "bounds": ["negative_infinity", "positive_infinity"],
        "residual": "nan",
    }
    metadata = error.to_metadata()
    assert metadata["application_family"] == "channel_pressure"
    assert metadata["stage"] == "pressure_solve"
    assert metadata["step"] == 0
    assert SimulationExecutionError.to_metadata(error)["details"] == dict(
        error.details
    )
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_plane_execution_error_finalizes_running_metadata(tmp_path):
    error = SimulationStateError(
        "Plane diagnostic is non-finite at step 0",
        application_family="plane_complete_stress",
        stage="diagnostic",
        step=0,
        details={"finite": False},
    )
    workflow = PlaneBerisEdwardsWorkflow(
        _FailingPlaneAdapter(error),
        _plane_spec(tmp_path),
        tmp_path,
        {"initial_condition": {}, "numerics": {}},
    )

    with pytest.raises(SimulationStateError) as captured:
        workflow.run()

    assert captured.value is error
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["completed_steps"] == 0
    assert metadata["workflow"]["failed_step"] == 0
    assert metadata["failure"] == error.to_metadata()
    assert not (tmp_path / "metadata.json.tmp").exists()
    assert not (tmp_path / "COMPLETE").exists()


def test_legacy_channel_pressure_error_finalizes_running_metadata(tmp_path):
    error = ChannelPressureConvergenceError(
        "production Channel pressure input contains NaN or Inf",
        details={"acceptable": False, "termination_reason": "nonfinite_input"},
    )
    spec = ChannelActiveNematicRunSpec(
        shape=(8, 6, 5),
        lengths=(8.0, 6.0, 5.0),
        steps=1,
        save_interval=1,
        diagnostic_interval=1,
        runtime_path=ChannelRuntimePath.LEGACY_CHANNEL,
    )
    workflow = ChannelActiveNematicsWorkflow(
        _FailingChannelAdapter(error),
        spec,
        tmp_path,
        {},
    )

    with pytest.raises(ChannelPressureConvergenceError) as captured:
        workflow.run()

    assert captured.value is error
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["completed_steps"] == 0
    assert metadata["workflow"]["failed_step"] == 0
    assert metadata["failure"] == error.to_metadata()
    assert not (tmp_path / "metadata.json.tmp").exists()
    assert not (tmp_path / "COMPLETE").exists()


@pytest.mark.parametrize(
    ("workflow_factory", "adapter_factory", "application_family"),
    (
        (
            lambda adapter, spec, path: PlaneBerisEdwardsWorkflow(
                adapter,
                spec,
                path,
                {"initial_condition": {}, "numerics": {}},
            ),
            _FailingPlaneAdapter,
            "plane_complete_stress_beris_edwards",
        ),
        (
            lambda adapter, spec, path: ChannelActiveNematicsWorkflow(
                adapter,
                spec,
                path,
                {},
            ),
            _FailingChannelAdapter,
            "legacy_channel_active_nematic",
        ),
    ),
)
def test_unexpected_workflow_failure_cannot_leave_running_metadata(
    tmp_path,
    workflow_factory,
    adapter_factory,
    application_family,
):
    error = RuntimeError("unexpected execution failure")
    if application_family.startswith("plane"):
        spec = _plane_spec(tmp_path)
    else:
        spec = ChannelActiveNematicRunSpec(
            shape=(8, 6, 5),
            lengths=(8.0, 6.0, 5.0),
            steps=1,
            save_interval=1,
            diagnostic_interval=1,
            runtime_path=ChannelRuntimePath.LEGACY_CHANNEL,
        )
    workflow = workflow_factory(adapter_factory(error), spec, tmp_path)

    with pytest.raises(RuntimeError) as captured:
        workflow.run()

    assert captured.value is error
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["failure"] == {
        "code": "simulation_execution_error",
        "message": "unexpected execution failure",
        "application_family": application_family,
        "stage": "workflow_execution",
        "step": 0,
        "details": {"legacy_exception_type": "RuntimeError"},
    }
