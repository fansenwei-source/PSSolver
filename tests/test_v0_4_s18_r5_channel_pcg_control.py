"""S-18 R5 shared Channel-PCG control regressions."""

from __future__ import annotations

import inspect

from benchmarks import verify_v0_4_s18_r5_pcg_equivalence as equivalence
import pssolver.functional.channel_pressure as channel_pressure
import pssolver.functional.channel_pressure_adjoint as channel_pressure_adjoint
import pssolver.linear_solvers.stokes as stokes_package
from pssolver.linear_solvers.stokes import channel_no_slip
from pssolver.linear_solvers.stokes._channel_pcg_control import (
    CHANNEL_PCG_FIXED_WORK_COMPLETED,
    CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION,
    CHANNEL_PCG_ITERATION_LIMIT,
    ChannelPCGControl,
)


def test_private_control_is_not_reexported_from_the_public_stokes_package():
    assert not hasattr(stokes_package, "ChannelPCGControl")
    assert not hasattr(stokes_package, "ChannelPCGStop")


def test_all_four_paths_use_the_shared_control_without_old_floor_state():
    sources = {
        "production": inspect.getsource(
            channel_no_slip.ChannelNoSlipModalStokesSolver._solve_pressure
        ),
        "functional_transpose": inspect.getsource(
            channel_pressure.ChannelPressureTransposeOperator.solve_pressure_transpose
        ),
        "implicit": inspect.getsource(
            channel_pressure_adjoint.ChannelImplicitPressureAdjoint._solve_no_grad
        ),
        "oracle": inspect.getsource(
            channel_pressure_adjoint.unrolled_channel_pressure_solve_oracle
        ),
    }
    for source in sources.values():
        assert "ChannelPCGControl(" in source
        assert "fixed_work_floor_pressure" not in source
        assert "fixed_work_floor_iterations" not in source
        assert "fixed_work_floor_relative_residual" not in source
    assert "stop_at_iteration_limit(" in sources["production"]
    assert "stop_at_iteration_limit(" in sources["functional_transpose"]
    assert "stop_at_iteration_limit(" in sources["implicit"]
    assert "stop_for_failure(" in sources["oracle"]


def test_floor_capture_keeps_only_the_smallest_recurrence_residual():
    control = ChannelPCGControl[str](fixed_work=True)
    assert control.should_capture_floor(
        represented_at_floor=True,
        relative_residual=0.25,
    )
    control.capture_floor(
        "first",
        iterations=3,
        relative_residual=0.25,
    )
    assert not control.should_capture_floor(
        represented_at_floor=True,
        relative_residual=0.5,
    )
    assert control.should_capture_floor(
        represented_at_floor=True,
        relative_residual=0.125,
    )
    control.capture_floor(
        "best",
        iterations=7,
        relative_residual=0.125,
    )
    assert control.floor_value == "best"
    assert control.floor_iterations == 7
    assert control.floor_relative_residual == 0.125


def test_breakdown_rolls_back_only_after_a_fixed_work_floor_exists():
    control = ChannelPCGControl[str](fixed_work=True)
    before_floor = control.stop_for_failure(
        "current",
        iterations=4,
        failure_reason="operator_breakdown",
    )
    assert before_floor.value == "current"
    assert before_floor.iterations == 4
    assert before_floor.termination_reason == "operator_breakdown"
    assert before_floor.acceptable is False
    assert before_floor.rolled_back_to_floor is False

    control.capture_floor(
        "floor",
        iterations=2,
        relative_residual=0.125,
    )
    after_floor = control.stop_for_failure(
        "current",
        iterations=9,
        failure_reason="operator_breakdown",
    )
    assert after_floor.value == "floor"
    assert after_floor.iterations == 2
    assert after_floor.termination_reason == (
        CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION
    )
    assert after_floor.acceptable is True
    assert after_floor.saturation_failure_reason == "operator_breakdown"
    assert after_floor.rolled_back_to_floor is True


def test_tolerance_and_fixed_work_iteration_limits_remain_distinct():
    tolerance = ChannelPCGControl[str](fixed_work=False)
    tolerance_stop = tolerance.stop_at_iteration_limit(
        "state",
        iterations=11,
    )
    assert tolerance_stop.termination_reason == CHANNEL_PCG_ITERATION_LIMIT
    assert tolerance_stop.acceptable is False

    fixed = ChannelPCGControl[str](fixed_work=True)
    fixed_stop = fixed.stop_at_iteration_limit("state", iterations=11)
    assert fixed_stop.termination_reason == CHANNEL_PCG_FIXED_WORK_COMPLETED
    assert fixed_stop.acceptable is True


def test_fixed_work_postcondition_fails_closed_with_original_reason():
    control = ChannelPCGControl[str](fixed_work=True)
    control.capture_floor(
        "floor",
        iterations=5,
        relative_residual=0.125,
    )
    control.stop_for_failure(
        "current",
        iterations=8,
        failure_reason="preconditioned_residual_breakdown",
    )
    acceptable, reason = control.finish_fixed_work_postcondition(
        acceptable=True,
        represented_at_floor=False,
        termination_reason=CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION,
        saturation_reason_still_at_floor=False,
    )
    assert acceptable is False
    assert reason == "preconditioned_residual_breakdown"

    acceptable, reason = control.finish_fixed_work_postcondition(
        acceptable=True,
        represented_at_floor=True,
        termination_reason=CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION,
        saturation_reason_still_at_floor=True,
    )
    assert acceptable is True
    assert reason == CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION


def test_equivalence_harness_covers_every_path_mode_and_dtype():
    assert equivalence.PATHS == (
        "production",
        "transpose",
        "implicit",
        "oracle",
    )
    cases = {row["name"]: row for row in equivalence.CASES}
    assert set(cases) == {
        "tolerance_float64",
        "fixed_float64",
        "fixed_float32_roundoff_floor",
        "fixed_float32_floor_crossing",
    }
    assert cases["tolerance_float64"]["fixed_iterations"] is None
    assert cases["fixed_float64"]["fixed_iterations"] == 5
    assert cases["fixed_float32_roundoff_floor"]["fixed_iterations"] == 50
    assert cases["fixed_float32_floor_crossing"]["fixed_iterations"] == 200
